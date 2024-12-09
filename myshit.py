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
    def __init__(self, shitcode: str, debug=False, parent: "MyShit" = None):
        self.shitcode = shitcode
        self.tokens = list(
            filter(lambda x: x != "", map(lambda x: x.strip(), shitcode.split()))
        )
        if self.tokens.count("::") > 0:
            raise ShitSyntaxError(":: is reserved symbol; your code cannot have it")
        self.cur = 0
        self.parsed_nodes = []
        self.debug = debug
        self.funcs = {}
        self.parent = parent

        self.vars: list[dict[str, Any]] = [dict()]
    def add_func(self, name, code):
        self.funcs[name] = code
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
            while self.try_token() != "EndShit":
                if self.try_token().startswith("%"):
                    func = self.get_token()[1:]
                    
                    func_name, func_args = func[:-1].split("(", 1)
                    func_args = func_args.split(",")
                    res = self.funcs.get(func_name)
                    if not res:
                        raise ShitSyntaxError("Function undefined")
                    
                    new_tree = MyShit(res.expand(func_args), self.debug, self)
                    res = new_tree.parse_shit()
                    continue
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
                            f"{var[0]}: {var[1].get_py_value() if var[1] else None}" for var in vars.items()
                        )
                        print(f"Last scope closed with vars {vars}")
                    return

                expr = ExprShittyToken.parse_self(self)
                res = expr.interpret()
                if isinstance(res, tuple):
                    if res[0] == "return":
                        return res[1]

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
    
    @staticmethod
    def copy_value(value:str, tree: MyShit):
        if value.isnumeric():
            return ValueShittyToken(value, tree)
        
        else:
            return ValueShittyToken("'" + value + "'", tree)

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
        elif eq_sign == "SHIT":
            tree.mv()
            arg_str = tree.try_token()
            if not arg_str.startswith("("):
                raise ShitSyntaxError("Function declaration does not have args")
            args = arg_str[1:-1].split(",")
            
            tree.mv()

            if tree.try_token() != "{":
                raise ShitSyntaxError("Function declaration does not have braces")
            
            tree.mv()
            commands = []
            while tree.try_token() != "}":
                cmd = tree.get_token()
                for arg in args:
                    if cmd == arg:
                        cmd = "::"+arg
                commands.append(cmd)
            tree.mv()
 
            func = ShitFunc(args, commands)
            tree.add_func(lhs, func)
            rhs = None
        else:
            rhs = None

        if rhs:
            tree.update_val(lhs, ValueShittyToken.copy_value(str(rhs.get_py_value()), tree))
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
                if self.tree.parent:
                    self.tree.parent.update_val("return", ValueShittyToken.copy_value(str(self.rhs.get_py_value()), self.tree.parent))
                    return ('return', self.rhs.get_py_value())
                else:
                    raise ShitSyntaxError("Чего ты сделать то пытаешься то")
            else:
                self.tree.vars[-2]["return"] = self.rhs


class ShitFunc:
    def __init__(self, args: list[str], tokens: list[str]):
        self.args = args
        self.tokens = tokens

    def expand(self, args: list[Any]):
        res = " "
        if isinstance(args, (list, tuple)):
            if len(self.args) != len(args):
                raise ShitSyntaxError("Argument count mismatch")
            for cmd in self.tokens:
                tokens = cmd.split(" ")
                _cmd = []
                for tok in tokens:
                    if "::" in tok:
                        arg_name = tok[2:]
                        idx = self.args.index(arg_name)
                        _cmd.append(args[idx])
                    else:
                        _cmd.append(tok)
                res += ' '.join(_cmd)+' '


        
        return res
    

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
