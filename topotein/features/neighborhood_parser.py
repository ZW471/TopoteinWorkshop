import torch
from graphein.protein.tensor.data import ProteinBatch
from lark import Lark, Transformer, v_args
from topomodelx.utils.sparse import from_sparse
from toponetx import CellComplex


# --- Helper functions ---


def parse_variable(var_name: str):
    """
    Given a variable name of the form <letter><rank> (where letter is one of
    Ad, Au, Ld, Lu, H, B, L, A and rank is 0, 1, or 2), split it into letter and rank.
    """
    if var_name.startswith(("Ad", "Au", "Ld", "Lu")):
        letter = var_name[:2]
        rank = int(var_name[2:])
    else:
        letter = var_name[0]
        rank = int(var_name[1:])
    return letter, rank

def create_sparse_tensor(letter: str, rank: int, cc: CellComplex, signed=False) -> torch.Tensor:
    """
    Create a torch sparse tensor based on the letter and rank.
      - Rank 0: a tensor of shape (1,)
      - Rank 1: a vector of length 2
      - Rank 2: a 2x2 matrix
    Different letters produce different hard-coded nonzero patterns.
    """
    if letter == "Ad":
        matrix = cc.adjacency_matrix(rank=rank, signed=signed)
    elif letter == "Au" or letter == "A":
        matrix = cc.coadjacency_matrix(rank=rank, signed=signed)
    elif letter == "Ld":
        matrix = cc.down_laplacian_matrix(rank=rank, signed=signed)
    elif letter == "Lu" or letter == "L":
        matrix = cc.up_laplacian_matrix(rank=rank, signed=signed)
    elif letter == "H":
        matrix = cc.hodge_laplacian_matrix(rank=rank, signed=signed)
    elif letter == "B":
        matrix = cc.incidence_matrix(rank=rank, signed=signed)
    else:
        raise ValueError(f"Unknown letter: {letter}")
    return from_sparse(matrix)

# --- Grammar definition ---
#
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

    VAR: /(Ad|Au|Ld|Lu|H|B|L|A)[0-2]/
    NUMBER: /\d+(\.\d+)?/
    FUNC: "inv"
    TRANSPOSE: ".T"

    %import common.WS
    %ignore WS
"""

# --- Transformer definition ---
#
# We use @v_args(inline=True) so that children are passed as arguments.
# Also, __default__ flattens any nodes that don’t have a specific method.
@v_args(inline=True)
class MatrixTransformer(Transformer):
    def __default__(self, data, children, meta):
        # If a node was not explicitly handled, return its single child (if present)
        if len(children) == 1:
            return children[0]
        return children

    def __init__(self, batch, signed=False):
        super().__init__()
        self.batch = batch
        self.signed = signed

    def start(self, expr):
        return expr

    def group(self, expr):
        return expr

    def var(self, token):
        var_name = str(token)
        letter, rank = parse_variable(var_name)
        return create_sparse_tensor(letter, rank, self.batch)

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
            return torch.inverse(ensure_dense(expr_value))
        else:
            raise ValueError(f"Unknown function: {func_name}")

# --- Parse and transform the expression ---
if __name__ == "__main__":
    batch = torch.load("/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_featurised_batch_edge_processed_simple.pt", weights_only=False)

    parser = Lark(grammar, parser="lalr")
    mt = MatrixTransformer(batch.sse_cell_complex, signed=False)
    expressions = [
        "B2.T @ B1.T / 2",
        "A0 + Ad0",
        "(H1 + Au1) @ B1.T"
    ]
    for expression in expressions:
        try:
            # Parse the expression into a parse tree.
            tree = parser.parse(expression)
            # Now fully transform the tree.
            result = mt.transform(tree)
            print("Expression:", expression)
            print("Result tensor:")
            print(result)
        except Exception as e:
            print(f"Error processing expression '{expression}': {e}")