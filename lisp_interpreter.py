# 1. [LEXER] tokenize(source) -> tokens
# 2. [PARSER] parse(tokens) -> abstract syntax tree
# 3. [INTERPRETER] evaluate(tree, frame) -> result
    # default frame = globalFrame

from exceptions import SchemeError, SchemeSyntaxError, SchemeEvaluationError, SchemeNameError
import re


# lexing and parsing

def tokenize(source):
    """
    splits an input string into tokens (left parens, right parens,
    other whitespace-separated values).
    Arguments:
        source (str): a string containing the code of a Scheme expression
    """
    pattern = r'\".*?\"|[()]|[^()\s]+'

    # source = '(define   DIAMETER 10)\n(define bip " beep boop")\n; A world is a boolean.'
    
    lines = source.split('\n')
    # lines = ['(define   DIAMETER 10)', '(define bip " beep boop")', '; A world is a boolean.']

    nocomments = [line[:line.index(';')] if ';' in line else line 
                  for line in lines]
    # nocomments = ['(define   DIAMETER 10)', '(define bip " beep boop")']

    allinone = ' '.join(nocomments)
    # allinone = '(define   DIAMETER 10) (define bip " beep boop")'

    tokens = re.findall(pattern, allinone)
    # tokens = ['(', 'define', 'DIAMETER', '10', ')', '(', 'define', 'bip', '" beep boop"', ')']
    
    return tokens


def parse(tokens):
    """
    parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    def parse_helper(index):
        if tokens[index] == '(':
            index += 1
            exp = [] # start a subexp
            
            while index < len(tokens) and tokens[index] != ')': # recursively append inner subexps to exp
                subexp, index = parse_helper(index)
                exp.append(subexp)
            return exp, index + 1
        
        elif tokens[index] == ')': # a ')' without first a '('
            raise SchemeSyntaxError
        
        elif tokens[index].startswith('"') and tokens[index].endswith('"'):
            stringval = tokens[index][1:-1]
            return StringLiteral(stringval), index + 1 # just the string value as a literal
            
        else: # is symbol or number
            try:
                return int(tokens[index]), index + 1
            except ValueError:
                try:
                    return float(tokens[index]), index + 1
                except ValueError:
                    return tokens[index], index + 1

    expression, next_index = parse_helper(0)
    if next_index != len(tokens):
        print("debug error: ", tokens, expression, next_index)
        raise SchemeSyntaxError
    return expression


# built-in functions

def not_func(args):
    """
    not func
    """
    if len(args) != 1:
        raise SchemeEvaluationError
    else:
        return not args[0]


def cons_func(args):
    """
    cons
    """
    if len(args) != 2:
        raise SchemeEvaluationError
    return Pair(args[0], args[1])


def car_func(args):
    """
    car
    """
    if len(args) != 1:
        raise SchemeEvaluationError
    if not isinstance(args[0], Pair):
        raise SchemeEvaluationError
    return args[0].car

        
def cdr_func(args):
    """
    cdr
    """
    if len(args) != 1:
        raise SchemeEvaluationError
    if not isinstance(args[0], Pair):
        raise SchemeEvaluationError
    return args[0].cdr


def make_list(elements):
    """
    makes list
    """
    if not elements:
        return []
    elif len(elements) == 1:
        return Pair(elements[0], [])
    else:
        return Pair(elements[0], make_list(elements[1:]))

    
def list_length(args):
    """
    gets length of lists
    """
    list_ = args[0]
    if list_ == []:
        return 0
    if not is_list_helper([list_]):
        raise SchemeEvaluationError
    else:
        return 1 + list_length([list_.cdr])
    

def is_list_helper(obj):
    """
    checks if list 
    """
    if obj[0] == []:
        return True
    elif isinstance(obj[0], Pair):
        if obj[0].cdr == []:
            return True
        else:
            return is_list_helper([obj[0].cdr])
    else:
        return False


def list_ref_helper(inp):
    """list indexing"""
    list_ = inp[0]
    index = inp[1]
    if not isinstance(index, int):
        raise SchemeEvaluationError
    if not isinstance(list_, Pair):
        raise SchemeEvaluationError
    if list_ == []:
        raise SchemeEvaluationError
    if (is_list_helper([list_.cdr]) and list_.cdr == []) \
        or not is_list_helper([list_.cdr]):
        if index == 0:
            return list_.car
        else:
            raise SchemeEvaluationError
    if index == 0:
        return list_.car
    else:
        return list_ref_helper([list_.cdr, index-1])


def copy_list(list_):
    """
    list_ (type Pair)
    returns deep copy of list_ without mutating input
    """
    # copy(list_.car) is just list_.car if immutable (int, string)
    # recursive call to copy_list(list_.car) if list_.car is also a Pair
    # copy(list_.cdr) is just list_.cdr if list_.cdr == []
    # recurisve call to copy_list(list_.cdr)
    # Pair(copy(list_.car), copy(list_.cdr))
    if list_ == []:
        return []
    
    # if isinstance(list_.car, Pair):
    #     copy_car = copy_list(list_.car)
    # else:
    #     copy_car = list_.car
    
    if isinstance(list_.cdr, Pair):
        copy_cdr = copy_list(list_.cdr)
    else:
        copy_cdr = []
    
    return Pair(list_.car, copy_cdr)


def append_lists(lists):
    """concatenates lists"""
    if len(lists) == 0:
        return []
    if len(lists) == 1:
        if is_list_helper([lists[0]]):
            if lists[0] == []:
                return []
            return Pair(lists[0].car, lists[0].cdr)
        else:
            raise SchemeEvaluationError
    else:
        new_list = None
        pointer = None
        for li in lists:
            if is_list_helper([li]):
                if li == []:
                    continue
                if new_list is None:
                    new_list = copy_list(li)
                    pointer = new_list
                else:
                    while pointer.cdr != []:
                        pointer = pointer.cdr
                    pointer.cdr = copy_list(li)
            else:
                raise SchemeEvaluationError
        if new_list:
            return new_list
        return []
        

def map_helper(args):
    """
    map
    """
    func = args[0]
    list_ = args[1]

    if list_ == []:
        return []
    
    if not isinstance(list_, Pair) or not callable(func):
        raise SchemeEvaluationError
    
    if list_.cdr == []: # bc: cons w/ 2 elems
        return Pair(func([list_.car]), list_.cdr)
    else: # rc 
        return Pair(func([list_.car]), map_helper([func, list_.cdr]))


def filter_helper(args):
    """
    filter
    """
    if len(args) != 2:
        raise SchemeEvaluationError
    
    func, list_ = args

    if list_ == []:
        return []

    if not isinstance(list_, Pair) or not callable(func):
        raise SchemeEvaluationError
    
    if list_.cdr == []: # bc: cdr is 'nil'
        if func([list_.car]):
            return Pair(list_.car, list_.cdr)
    
    if func([list_.car]): # if func passes car
        return Pair(list_.car, filter_helper([func, list_.cdr]))
    else: # if func doesn't pass car. try again w/ cdr
        return filter_helper([func, list_.cdr])


def reduce_helper(args):
    """
    reduce
    """
    func = args[0]
    list_ = args[1]
    initval = args[2]
    
    if list_ == []:
        return initval
    
    if not isinstance(list_, Pair) or not callable(func):
        raise SchemeEvaluationError
    
    if list_.cdr == []:
        return func([initval, list_.car])
    else:
        return reduce_helper([func, list_.cdr, func([initval, list_.car])])


def begin_helper(args):
    """
    begin
    expressions are evaluated sequentially from left to right, and the value of the last expression is returned. 
    This expression type is used to sequence side effects such as input and output
    """
    return args[-1]


scheme_builtins = {
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 \
         else (args[0] - sum(args[1:])),
    '*': lambda args: args[0] if len(args) == 1 \
         else (args[0] * scheme_builtins['*'](args[1:])),
    '/': lambda args: args[0] if len(args) == 1 \
         else (args[0] / scheme_builtins['*'](args[1:])),
    '#t': True,
    '#f': False,
    'nil': [],
    
    'equal?': lambda args: False if False in [args[i] == args[i+1] \
                           for i in range(len(args)-1)] else True,
    '>':      lambda args: False if False in [args[i] > args[i+1] \
                           for i in range(len(args)-1)] else True,
    '>=':     lambda args: False if False in [args[i] >= args[i+1] \
                           for i in range(len(args)-1)] else True,
    '<':      lambda args: False if False in [args[i] < args[i+1] \
                           for i in range(len(args)-1)] else True,
    '<=':     lambda args: False if False in [args[i] <= args[i+1] \
                           for i in range(len(args)-1)] else True,
    'list': make_list,
    'not':  not_func,
    'cons': cons_func,
    'car':  car_func,
    'cdr':  cdr_func,

    'list?': is_list_helper,
    'length': list_length,
    'list-ref': list_ref_helper,
    'append': append_lists,
    'map': map_helper,
    'filter': filter_helper,
    'reduce': reduce_helper,

    'begin': begin_helper,
}


# evaluation

class Frame:
    """
    frames
    """
    def __init__(self, parent=None):
        self.vars = {}
        self.parent = parent

    def __setitem__(self, set_var, val):
        self.vars[set_var] = val
        
    def __getitem__(self, get_var):
        if get_var in self.vars:
            return self.vars[get_var]
        if self.parent is not None: # recursively look in parent frame
            return self.parent[get_var]
        raise SchemeNameError(f"variable {get_var} not defined")
    
    def __delitem__(self, var):
        if var in self.vars:
            x = self.vars[var]
            del self.vars[var]
            return x
        else:
            raise SchemeNameError(f"variable {var} not defined")
        
    def setbang(self, var, expr): # reassign var to expr
        if var in self.vars:
            self.vars[var] = expr
            return expr
        else:
            if self.parent is not None:
                return self.parent.setbang(var, expr)
            raise SchemeNameError(f"variable {var} not defined")

# add builtins to globalFrame
globalFrame = Frame()
for k, v in scheme_builtins.items():
    globalFrame[k] = v

class Function:
    """
    lambda functions
    """
    def __init__(self, parameters, body, frame):
        self.parameters = parameters
        self.body = body
        self.frame = frame

    def __call__(self, args):
        if len(self.parameters) != len(args):
            raise SchemeEvaluationError
        mappings = {self.parameters[i]:args[i] for i in range(len(self.parameters))}
        parent_frame = Frame(self.frame)
        parent_frame.vars = mappings
        return evaluate(self.body, parent_frame)


class Pair:
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def __str__(self):
        if self.cdr == []:
            return '[' + str(self.car) + ',' + '[]]'
        else:
            return '[' + str(self.car) + ',' + str(self.cdr) + ']'


"""
Python dynamic type checks so an object cast as StringLiteral would also be str at runtime
"""
# class StringLiteral(str):
#     def __new__(cls, text):
#         return super().__new__(cls, text)

#     def __init__(self, text):
#         self.text = text

class StringLiteral:
    def __init__(self, text):
        self.text = text
    

def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if frame is None:
        frame = Frame(globalFrame)

    if type(tree) in [float, int]: # number
        return tree
    
    elif isinstance(tree, StringLiteral): # string
        return tree.text
    
    elif isinstance(tree, str): # symbol
        return frame[tree]

    if not isinstance(tree, list): # should be list
        raise SchemeEvaluationError
    
    if not tree:
        raise SchemeEvaluationError
    
    operator = tree[0]

    if operator == 'define':
        # variable declaration
        if not isinstance(tree[1], list):
            if not isinstance(tree[1], str):
                raise SchemeEvaluationError("bad identifier")
                
            var = tree[1]
            val = evaluate(tree[2], frame)
            frame[var] = val
            return val
        # function declaration
        else:
            # turn this into variable declaration by setting function body as a lambda function to function/'variable' name
            new_exp = ['define', tree[1][0], ['lambda', tree[1][1:], tree[2]]]
            return evaluate(new_exp, frame)
        
    if operator == 'lambda':
        func = Function(tree[1], tree[2], frame) # Function(args, body, frame)
        return func
    
    if operator == 'if':
        is_pred = evaluate(tree[1], frame) # t or f

        if is_pred:
            return evaluate(tree[2], frame)
        else:
            return evaluate(tree[3], frame)

    if operator == 'cond':
        for clause in tree[1:]:
            pred, exprs = clause[0], clause[1:]
            if pred == 'else' or evaluate(pred, frame):
                val = None
                for expr in exprs:
                    val = evaluate(expr, frame)
                return val
        return None
    
    if operator == 'case':
        key = evaluate(tree[1], frame)
        for clause in tree[2:]:
            objs, exprs = clause[0], clause[1:]
            if objs == 'else' or key in objs:
                val = None
                for expr in exprs:
                    val = evaluate(expr, frame)
                return val
        return None

    if operator == 'and': # short circuit at first false
        for arg in tree[1:]:
            if evaluate(arg, frame) is False:
                return False
        return True

    if operator == 'or': # short circuit at first true
        for arg in tree[1:]:
            if evaluate(arg, frame) is True:
                return True
        return False
    
    if operator == 'del':
        return frame.__delitem__(tree[1])

    if operator == 'let':
        new_frame = Frame(frame)
        for var, val in tree[1]:
            new_frame[var] = evaluate(val, frame)
        return evaluate(tree[2], new_frame)

    if operator == 'set!':
        new_exp = evaluate(tree[2], frame)
        return frame.setbang(tree[1], new_exp)

    args = []
    for _, item in enumerate(tree): # adds all the fxn's args to the list
        arg = evaluate(item, frame)
        args.append(arg)
    func = args[0]
    params = args[1:]
    if callable(func):
        return func(params)
    raise SchemeEvaluationError


def evaluate_file(file_name, frame=None):
    """
    evaluates file contents
    """
    if frame is None:
        frame = Frame(globalFrame)
    with open(file_name, 'r', encoding='utf-8') as file:
        contents = file.read() #returns a string w all the contents of a file
    return evaluate(parse(tokenize(contents)), frame)


def repl(verbose=False):
    """
    read 1 line of user input, evaluate the expression, print 
    out the result. (if verbose=True, will also print tokens and parsed expression)
    repeat until user inputs 'QUIT'
    """
    import traceback
    _, frame = result_and_frame(['+'])  # make a global frame

    while True:
        input_str = input('in> ')
        if input_str == 'QUIT':
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print('tokens>', token_list)
            expression = parse(token_list)
            if verbose:
                print('expression>', expression)
            output, frame = result_and_frame(expression, frame)
            print('  out>', output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print('Error>', repr(e))


def result_and_frame(tree, frame=None):
    if frame is None:
        frame = Frame(globalFrame)
    return evaluate(tree, frame), frame
    

if __name__ == '__main__':
    repl(verbose=True)
