from src.lib import lex

tokens = (
    "INT",
    "FLOAT",
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "QUO",
    "REM",
    "NAME",
    "ASSIGN",
    "COMPARISON",
    "LPAREN",
    "RPAREN",
    "AND",
    "OR",
)

t_ignore = " \t"
t_ignore_COMMENT = r"\#.*"

t_ADD = r"\+"
t_SUB = r"-"
t_MUL = r"\*"
t_DIV = r"\/"
t_QUO = r"\/\/"
t_REM = r"\%"
t_COMPARISON = r"[=<>!]=|[<>]"
t_ASSIGN = r"=|\+=|-=|\*=|\/=|\/\/=|\%="
t_LPAREN = r"\("
t_RPAREN = r"\)"


def t_AND(t):
    r"and"
    return t


def t_OR(t):
    r"or"
    return t


def t_NAME(t):
    r"[a-zA-Z_][a-zA-Z0-9_]*"
    return t


def t_FLOAT(t):
    r"([0-9]+\.[0-9]*)"
    t.value = float(t.value)
    return t


def t_INT(t):
    r"([1-9][0-9]*)|0"
    t.value = int(t.value)
    return t


def t_newline(t):
    r"""\n+"""
    t.lexer.lineno += t.value.count("\n")


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


lex.lex()
