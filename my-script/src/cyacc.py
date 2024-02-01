from src.clex import tokens
from src.lib import yacc

# Precedence rules for the arithmetic operators
precedence = (
    ("left", "COMPARISON"),
    ("left", "AND", "OR"),
    ("left", "ADD", "SUB"),
    ("left", "MUL", "DIV", "QUO", "REM"),
    ("right", "UMINUS"),
)

names = {}


def p_statement_assign(p):
    "statement : NAME ASSIGN expression"
    if p[2] == "=":
        names[p[1]] = p[3]
    else:
        try:
            if p[2] == "+=":
                names[p[1]] += p[3]
            if p[2] == "-=":
                names[p[1]] -= p[3]
            if p[2] == "*=":
                names[p[1]] *= p[3]
            if p[2] == "/=":
                names[p[1]] /= p[3]
            if p[2] == "//=":
                names[p[1]] //= p[3]
            if p[2] == "%=":
                names[p[1]] %= p[3]
        except LookupError:
            print(f"Undefined name {p[1]!r}")


def p_statement_expr(p):
    "statement : expression"
    print(">> " + str(p[1]))


def p_expression_comparison(p):
    """expression : expression COMPARISON expression"""
    if p[2] == "==":
        p[0] = int(p[1] == p[3])
    if p[2] == "!=":
        p[0] = int(p[1] != p[3])
    if p[2] == ">=":
        p[0] = int(p[1] >= p[3])
    if p[2] == "<=":
        p[0] = int(p[1] <= p[3])
    if p[2] == ">":
        p[0] = int(p[1] > p[3])
    if p[2] == "<":
        p[0] = int(p[1] < p[3])


def p_expression_binop(p):
    """expression : expression ADD expression
    | expression SUB expression
    | expression MUL expression
    | expression DIV expression
    | expression QUO expression
    | expression REM expression
    | expression AND expression
    | expression OR expression"""
    if p[2] == "+":
        p[0] = p[1] + p[3]
    elif p[2] == "-":
        p[0] = p[1] - p[3]
    elif p[2] == "*":
        p[0] = p[1] * p[3]
    elif p[2] == "/":
        p[0] = p[1] / p[3]
    elif p[2] == "//":
        p[0] = p[1] // p[3]
    elif p[2] == "%":
        p[0] = p[1] % p[3]
    elif p[2] == "and":
        p[0] = p[1] and p[3]
    elif p[2] == "or":
        p[0] = p[1] or p[3]


def p_expression_uminus(p):
    "expression : SUB expression %prec UMINUS"
    p[0] = -p[2]


def p_expression_group(p):
    "expression : LPAREN expression RPAREN"
    p[0] = p[2]


def p_expression_number(p):
    """expression : FLOAT
    | INT"""
    p[0] = p[1]


def p_expression_name(p):
    "expression : NAME"
    try:
        p[0] = names[p[1]]
    except LookupError:
        print(f"Undefined name {p[1]!r}")
        p[0] = 0


def p_error(p):
    print(f"Syntax error at {p.value!r}")


yacc.yacc()


def parse(data):
    return yacc.parse(data)
