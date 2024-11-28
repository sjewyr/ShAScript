import sys
from typing import Any


class ShitLexicError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"Lexic Error: {self.msg}"


class ShitSyntaxError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return f"Syntax Error: {self.msg}"


class MyShit:
    def __init__(self, shitcode: str, debug=False):
        self.shitcode = shitcode
        self.tokens = list(
            filter(lambda x: x != "", map(lambda x: x.strip(), shitcode.split()))
        )
        self.cur = 0
        self.parsed_nodes = []
        self.debug = debug

        self.vars: list[dict[str, Any]] = [dict()]

    def append_var(self, name, val):
        self.vars[-1][name] = val

    def try_get_val(self, name):
        return self.vars[-1].get(name)

    def update_val(self, name, val):
        self.vars[-1][name] = val

    def parse_shit(self):
        if self.tokens.count("{") != self.tokens.count("}"):
            raise ShitSyntaxError("Scope Mismatch")
        try:
            while self.try_token() != "OhShit":
                self.parsed_nodes.append(VarShittyToken.parse_self(self))
        except IndexError:
            raise ShitLexicError("No OhShit at the end of variable stream")

        self.mv()
        try:
            while self.try_token() != "EndShit":
                if self.try_token() == "{":
                    self.vars.append(dict())
                    self.mv()
                    if self.debug:
                        print("Opened new scope")

                    self.parse_shit()
                    continue
                if self.try_token() == "}":
                    vars = self.vars.pop()
                    self.mv()
                    if self.debug:
                        vars = "\n".join(
                            f"{var[0]}: {var[1].get_py_value()}" for var in vars.items()
                        )
                        print(f"Last scope closed with vars {vars}")
                    return

                expr = ExprShittyToken.parse_self(self)
                expr.interpret()

                self.parsed_nodes.append(expr)
        except IndexError:
            raise ShitLexicError("No EndShit at the end of expr stream")

        if len(self.vars) > 1:
            raise ShitSyntaxError("EndShit is not allowed outside global scope")
        self.mv()

        if self.debug:
            print(self)

    def get_token(self):
        cur = self.cur
        self.cur += 1
        return self.tokens[cur]

    def try_token(self):
        return self.tokens[self.cur]

    def mv(self):
        self.cur += 1

    def __str__(self):
        return "\n".join([str(node) for node in self.parsed_nodes])


class ShittyToken:
    def __init__(self, tree):
        self.tree = tree

    def parse_self(self, tree: MyShit):
        pass


class ValueShittyToken(ShittyToken):
    def __init__(self, value, tree):
        self.val = value
        self.tree = tree

    @staticmethod
    def parse_self(tree: MyShit):
        return ValueShittyToken(tree.get_token(), tree)

    def get_py_value(self):
        val = str(self.val)
        if val.startswith("'") and val.endswith("'"):
            val = str(val[1:-1])
        elif val.isnumeric():
            val = int(val)
        else:
            val = self.tree.try_get_val(self.val)
            if val:
                val = val.get_py_value()
        return val

    def __str__(self):
        return f"Value: {self.val}"


class ExprShittyToken(ShittyToken):
    def __init__(self, lhs, rhs, tree: MyShit):
        self.lhs = lhs
        self.rhs = rhs
        self.tree = tree

    @staticmethod
    def parse_self(tree: MyShit):
        lhs = tree.get_token()
        eq_sign = tree.try_token()
        if eq_sign == "=":
            tree.mv()
            rhs = ValueShittyToken.parse_self(tree)
        elif eq_sign == "@=":
            tree.mv()
            rhs = RHS.parse_self(tree)
        else:
            rhs = None

        if rhs:
            tree.update_val(lhs, ValueShittyToken(str(rhs.get_py_value()), tree))
        else:
            tree.update_val(lhs, None)
        return ExprShittyToken(lhs, rhs, tree)

    def __str__(self):
        return f"Expr: {self.lhs}, {self.rhs}"

    def interpret(self):
        if self.lhs == "print":
            print(self.rhs.get_py_value(), sep="")
        if self.lhs == "return":
            if len(self.tree.vars) == 1:
                raise ShitSyntaxError("Cannot return from global scope")
            self.tree.vars[-2]["return"] = self.rhs


class RHS:
    def __init__(self, tree):
        pass

    @staticmethod
    def parse_self(tree: MyShit) -> ValueShittyToken:
        val = MultBlock.parse_self(tree)
        if not isinstance(val, int):
            raise ShitSyntaxError("I cannot add non-numeric values")
        cur = tree.try_token()
        while cur == "+" or cur == "-":
            tree.mv()
            if cur == "+":
                val += MultBlock.parse_self(tree)
            else:
                val -= MultBlock.parse_self(tree)
            cur = tree.try_token()
        return ValueShittyToken(str(val), tree)


class MultBlock:
    @staticmethod
    def parse_self(tree: MyShit) -> int:
        val = ValueShittyToken.parse_self(tree).get_py_value()
        if not isinstance(val, int):
            raise ShitSyntaxError("I cannot multiply non-numeric values")

        cur = tree.try_token()
        while cur == "*" or cur == "/":
            tree.mv()
            _val = ValueShittyToken.parse_self(tree).get_py_value()
            if not isinstance(_val, int):
                raise ShitSyntaxError("I cannot multiply non-numeric values")
            if cur == "*":
                val *= _val
            else:
                val = val // _val
            cur = tree.try_token()

        return val


class VarShittyToken(ShittyToken):
    def __init__(self, name, tree, val=None):
        self.name = name
        self.val = val
        self.tree = tree

    @staticmethod
    def parse_self(tree: MyShit):
        name = tree.get_token()
        eq_sign = tree.try_token()
        val = None
        if eq_sign == "=":
            tree.mv()
            val = ValueShittyToken.parse_self(tree)

        tree.append_var(name, val)
        return VarShittyToken(name, tree, val)

    def __str__(self):
        return f"Var: {self.name}: {self.val}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        shit = sys.argv[1]
        if not shit.endswith(".shittyass"):
            print("Pass a file with a .shittyass extension")
        else:
            with open(shit, "r") as shit:
                shit = shit.read()
            if len(sys.argv) > 2:
                shit = MyShit(shit, True)
            else:
                shit = MyShit(shit)

            shit.parse_shit()
    else:
        print("Pass a file name to interpret")
