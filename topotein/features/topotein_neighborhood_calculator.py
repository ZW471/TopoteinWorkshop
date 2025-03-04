import torch
from lark import Lark, Transformer, v_args

from topotein.features.topotein_complex import TopoteinComplex


# --- Helper functions ---

def parse_variable(var_name: str):
    """
    Given a variable name in the format:
      - For Laplacian matrices: Lx_y, where x and y are digits from 0 to 2 (and x != y)
      - For Adjacency matrices: Ax_y, where x and y are digits from 0 to 2
      - For Incidence matrices: Bx_y, where x and y are digits from 0 to 2
    This function extracts and returns:
      - letter: one of "A", "B", or "L"
      - rank1: the first digit as an integer
      - rank2: the second digit as an integer
    """
    if len(var_name) != 4 or var_name[2] != '_':
        raise ValueError(f"Invalid variable format: {var_name}. Expected format: [A|B|L][0-2]_[0-2]")
    letter = var_name[0]
    rank1 = int(var_name[1])
    rank2 = int(var_name[3])
    return letter, rank1, rank2

def create_sparse_tensor(letter: str, rank1: int, rank2: int, complex_instance, device="cpu") -> torch.Tensor:
    """
    Create a torch sparse tensor based on the variable type and rank numbers,
    using the provided TopoteinComplex instance.

    For:
      - "A": calls the complex_instance.adjacency_matrix(rank=rank1, via_rank=rank2)
      - "B": calls the complex_instance.incidence_matrix(from_rank=rank1, to_rank=rank2)
      - "L": calls the complex_instance.laplacian_matrix(rank=rank1, via_rank=rank2)
    """
    if letter == "A":
        matrix = complex_instance.adjacency_matrix(rank=rank1, via_rank=rank2)
    elif letter == "B":
        matrix = complex_instance.incidence_matrix(from_rank=rank1, to_rank=rank2)
    elif letter == "L":
        matrix = complex_instance.laplacian_matrix(rank=rank1, via_rank=rank2)
    else:
        raise ValueError(f"Unknown matrix type: {letter}")
    return matrix.to(device)

# --- Grammar definition ---
#
# The only change here is the VAR token, which now only accepts variables
# of the form A|B|Lx_y (where x and y are in [0-2]).
grammar = r"""
    start: expr

    expr: expr "+" term   -> add
        | expr "-" term   -> sub
        | term

    term: term "@" factor  -> matmul
        | term "/" factor  -> div
        | factor

    factor: primary TRANSPOSE?  -> transpose

    primary: FUNC "(" expr ")"  -> func_call
           | VAR                -> var
           | NUMBER             -> number
           | "(" expr ")"       -> group

    VAR: /(A|B|L)[0-2]_[0-2]/
    NUMBER: /\d+(\.\d+)?/
    FUNC: "inv"
    TRANSPOSE: ".T"

    %import common.WS
    %ignore WS
"""

# --- Transformer definition ---
#
# We use the inline arguments so that children are passed as arguments.
@v_args(inline=True)
class MatrixTransformer(Transformer):
    def __default__(self, data, children, meta):
        # If a node was not explicitly handled, return its single child (if present)
        if len(children) == 1:
            return children[0]
        return children

    def __init__(self, complex_instance, use_cache=True, device="cpu"):
        super().__init__()
        self.complex_instance = complex_instance
        self.use_cache = use_cache
        self.device = device
        self._cache = {}

    def start(self, expr):
        return expr

    def group(self, expr):
        return expr

    def var(self, token):
        var_name = str(token)
        if self.use_cache and var_name in self._cache:
            return self._cache[var_name]
        letter, rank1, rank2 = parse_variable(var_name)
        matrix = create_sparse_tensor(letter, rank1, rank2, self.complex_instance, device=self.device)
        if self.use_cache:
            self._cache[var_name] = matrix
        return matrix

    def number(self, token):
        return float(token)

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def matmul(self, a, b):
        return torch.sparse.mm(a, b)

    def div(self, a, b):
        if isinstance(b, (int, float)):
            return a / b
        else:
            raise ValueError("Division supports only scalar division.")

    def transpose(self, value, t_suffix=None):
        if t_suffix is not None:
            if isinstance(value, torch.Tensor) and value.dim() == 2:
                return value.transpose(0, 1)
        return value

    def func_call(self, func_name, expr_value):
        func_name = str(func_name)
        if func_name == "inv":
            return torch.inverse(expr_value)
        else:
            raise ValueError(f"Unknown function: {func_name}")

# --- Calculator class ---
#
# This class ties together the parser and the transformer. It expects a
# TopoteinComplex instance and evaluates expressions (or full equations)
# based on the new variable naming scheme.
class TopoteinNeighborhoodCalculator:
    def __init__(self, topotein_complex, use_cache=True):
        self.transformer = MatrixTransformer(topotein_complex, use_cache=use_cache, device=topotein_complex.device)
        self.parser = Lark(grammar, parser="lalr")

    def eval(self, expression):
        tree = self.parser.parse(expression)
        return self.transformer.transform(tree).coalesce()

    def equation_to_expression(self, equation):
        variable, expression = equation.split("=")
        variable = variable.strip()
        expression = expression.strip()
        return variable, expression

    def calc_equations(self, equations):
        results = {}
        for equation in equations:
            variable, expression = self.equation_to_expression(equation)
            try:
                results[variable] = self.eval(expression)
            except Exception as e:
                print(f"Error processing expression '{expression}': {e}")
        return results

# --- Example usage ---
#
# Here we create a dummy TopoteinComplex and evaluate a few equations.
if __name__ == "__main__":
    # Dummy data for demonstration (in practice, use your actual batch data)
    device = "cpu"
    num_nodes = 10
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device=device)
    cell_index = torch.tensor([[0, 6], [3, 9]], dtype=torch.long, device=device)  # example: two cells

    complex_instance = TopoteinComplex(num_nodes, edge_index, cell_index, use_cache=True)
    calculator = TopoteinNeighborhoodCalculator(complex_instance)
    equations = [
        "N0_1_from_B = B0_1 @ B0_1.T / 2",
        "N0_1_from_L = L0_1",
        "B1_2 = B1_2",
        "N0_2 = B0_2"
    ]
    import time
    tik = time.time()
    results = calculator.calc_equations(equations)
    print(f"Time taken: {time.time() - tik} seconds")
    print(results)