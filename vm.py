"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp

from typing import Any

CO_VARARGS = 4
CO_VARKEYWORDS = 8

ERR_TOO_MANY_POS_ARGS = "Too many positional arguments"
ERR_TOO_MANY_KW_ARGS = "Too many keyword arguments"
ERR_MULT_VALUES_FOR_ARG = "Multiple values for arguments"
ERR_MISSING_POS_ARGS = "Missing positional arguments"
ERR_MISSING_KWONLY_ARGS = "Missing keyword-only arguments"
ERR_POSONLY_PASSED_AS_KW = "Positional-only argument passed as keyword argument"


def bind_args(
    code: Any,
    default_values: tuple[(Any, ...)],
    kw_default: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bind values from `args` and `kwargs` to corresponding arguments of `func`

    :param func: function to be inspected
    :param args: positional arguments to be bound
    :param kwargs: keyword arguments to be bound
    :return: `dict[argument_name] = argument_value` if binding was successful,
             raise TypeError with one of `ERR_*` error descriptions otherwisefsd
    """
    result: dict[str, Any] = {}
    flag = code.co_flags
    variables = code.co_varnames[
                : code.co_argcount
                + code.co_kwonlyargcount
                + (flag & CO_VARARGS > 0)
                + (flag & CO_VARKEYWORDS > 0)
            ]
    pos_count = code.co_argcount
    pos_only_count = code.co_posonlyargcount
    kw_only_count = code.co_kwonlyargcount

    pos_only = variables[:pos_only_count]
    pos = variables[:pos_count]
    kw_only = variables[pos_count: pos_count + kw_only_count]

    pos_default = {
        arg: value for arg, value in zip(pos[-len(default_values):], default_values)
    }

    if flag & CO_VARARGS:
        args_name = variables[-1 - (flag & CO_VARKEYWORDS > 0)]
        result[args_name] = []

    if flag & CO_VARKEYWORDS:
        kwargs_name = variables[-1]
        result[kwargs_name] = {}

    # parse positional only arguments

    for i in range(pos_only_count):
        arg = pos_only[i]
        if arg in kwargs and not (flag & CO_VARKEYWORDS):
            raise TypeError(ERR_POSONLY_PASSED_AS_KW)
        if i >= len(args):
            if arg in pos_default:
                result[arg] = pos_default[arg]
            else:
                raise TypeError(ERR_MISSING_POS_ARGS)
        else:
            result[arg] = args[i]

    # parse positional arguments

    if len(args) > len(pos) and not (flag & CO_VARARGS):
        raise TypeError(ERR_TOO_MANY_POS_ARGS)
    for i in range(pos_only_count, min(pos_count, len(args))):
        arg = pos[i]
        result[arg] = args[i]

    # parse keyword only arguments

    for i in range(kw_only_count):
        arg = kw_only[i]
        if arg not in kwargs:
            if arg in kw_default:
                result[arg] = kw_default[arg]
            else:
                raise TypeError(ERR_MISSING_KWONLY_ARGS)
        else:
            if arg in result:
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)
            result[arg] = kwargs[arg]

    # parse keyword arguments

    for arg, value in kwargs.items():
        if arg not in variables or arg in kw_only:
            continue
        if arg in result:
            if flag & CO_VARKEYWORDS and arg in pos_only:
                result[kwargs_name][arg] = value
            else:
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)
        else:
            if arg in result:
                raise TypeError(ERR_MULT_VALUES_FOR_ARG)
            result[arg] = value

    # parse CO_VARARGS and CO_VARKEYWORDS

    if flag & CO_VARARGS:
        result[args_name] = args[len(pos):]

    if flag & CO_VARKEYWORDS:
        for arg, value in kwargs.items():
            if arg not in variables:
                result[kwargs_name][arg] = value

    # parse default values

    for item in variables:
        if item not in result:
            if item in pos_default:
                result[item] = pos_default[item]
            elif item in kw_default:
                result[item] = kw_default[item]
            else:
                raise TypeError(ERR_MISSING_POS_ARGS)

    return result


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.12/Include/internal/pycore_frame.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(
        self,
        frame_code: types.CodeType,
        frame_builtins: dict[str, tp.Any],
        frame_globals: dict[str, tp.Any],
        frame_locals: dict[str, tp.Any],
    ) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.return_check = False
        self.instruction_num = 0
        self.check_kwargs = False

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.dsfjak
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        instructions = list(dis.get_instructions(self.code))
        dct = {instruction.offset: instruction for instruction in instructions}
        deltas = {
            instructions[i]: instructions[i + 1].offset - instructions[i].offset
            for i in range(len(instructions) - 1)
        }
        while self.instruction_num in dct:
            instruction = dct[self.instruction_num]
            # print(instruction.offset, instruction.opname, self.data_stack)
            self.instruction_num += deltas.get(instruction, 1)
            getattr(self, instruction.opname.lower() + "_op")(instruction.argval)
            if self.return_check:
                return self.return_value
        return self.return_value

    # jump

    def jump_backward_op(self, delta: int) -> None:
        self.instruction_num = delta

    def jump_forward_op(self, delta: int) -> None:
        self.instruction_num = delta

    def pop_jump_if_true_op(self, delta: int) -> None:
        if self.pop():
            self.instruction_num = delta

    def pop_jump_if_none_op(self, delta: int) -> None:
        if self.pop() is None:
            self.instruction_num = delta

    def pop_jump_if_false_op(self, delta: int) -> None:
        if not self.pop():
            self.instruction_num = delta

    # base operations

    def kw_names_op(self, names: list[str]) -> None:
        result = {}
        for name, value in zip(names, self.popn(len(names))):
            result[name] = value
        self.push(result)
        self.check_kwargs = True

    def call_intrinsic_1_op(self, intrinsic_id: int) -> None:
        """
        Implements the CALL_INTRINSIC_1 operation, which calls an intrinsic function
        with one argument. The argument is expected to be on the stack, and the result
        replaces the argument after the function is called.

        The intrinsic_id determines which intrinsic function to call.
        """
        arg: Any = self.pop()
        if intrinsic_id == 1:
            print(arg)
            self.push(None)
        elif intrinsic_id == 2:
            module_dict = vars(arg)
            for name, value in module_dict.items():
                if not name.startswith("_"):
                    self.globals[name] = value
            self.push(None)
        elif intrinsic_id == 3:
            self.push(arg.value)
        elif intrinsic_id == 4:
            self.push(arg)
        elif intrinsic_id == 5:
            self.push(+arg)
        elif intrinsic_id == 6:
            self.push(tuple(arg))
        elif intrinsic_id == 7:
            self.push(f"TypeVar({arg})")
        elif intrinsic_id == 8:
            self.push(f"ParamSpec({arg})")
        elif intrinsic_id == 9:
            self.push(f"TypeVarTuple({arg})")
        elif intrinsic_id == 10:
            self.push(f"Generic[{arg}]")
        else:
            name, type_params, value = arg
            self.push(f"TypeAlias({name}, {type_params}, {value})")

    def nop_op(self, _: Any) -> None:
        pass

    def format_value_op(self, flag: tuple[Any, bool]) -> None:
        value = self.pop()
        frm_spec = self.pop() if flag[1] else ""
        if flag[0]:
            self.push(format(flag[0](value), frm_spec))
        else:
            self.push(format(value, frm_spec))

    def copy_op(self, i: int) -> None:
        self.push(self.data_stack[-i])

    def unpack_sequence_op(self, count: int) -> None:
        assert len(self.data_stack[-1]) == count
        self.data_stack.extend(self.pop()[::-1])

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-CALL
        """
        kwargs = {}
        if isinstance(self.data_stack[-1], dict) and self.check_kwargs:
            kwargs = self.pop()
        arguments = self.popn(arg - len(kwargs))
        f = self.pop()
        if self.data_stack and self.data_stack[-1] is None:
            self.pop()
        self.push(f(*arguments, **kwargs))
        self.check_kwargs = False

    def call_function_ex_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-CALL
        """
        kwargs = {}
        if arg == 0:
            arguments = self.pop()
        else:
            kwargs = self.pop()
            arguments = self.pop()
        f = self.pop()
        if self.data_stack and self.data_stack[-1] is None:
            self.pop()
        self.push(f(*arguments, **kwargs))
        self.check_kwargs = False

    # load

    def load_fast_check_op(self, var_name: str) -> None:
        if var_name in self.locals:
            self.push(self.locals[var_name])
        else:
            raise UnboundLocalError()

    def load_fast_and_clear_op(self, var_name: str) -> None:
        if var_name in self.locals:
            value = self.locals[var_name]
            self.push(value)
            self.locals[var_name] = None
        else:
            self.push(None)

    def load_attr_op(self, attr_name: str) -> None:
        self.push(getattr(self.pop(), attr_name))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization
        Partial realization

        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f"name '{arg}' is not defined")

    def load_global_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError(f"name '{arg}' is not defined")

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def load_fast_op(self, arg: str) -> None:
        self.push(self.locals[arg])

    # return

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()
        self.return_check = True

    def return_const_op(self, arg: tp.Any) -> None:
        self.return_value = arg
        self.return_check = True

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def make_function_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-MAKE_FUNCTION
        """

        # TODO: use arg to parse function defaults
        kwdefaults = {}
        defaults = ()
        code = self.pop()
        if arg & 0x02:
            kwdefaults = self.pop()
        if arg & 0x01:
            defaults = self.pop()

        def f(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            # TODO: parse input arguments using code attributes such as co_argcount

            parsed_args: dict[str, tp.Any] = bind_args(
                code, defaults, kwdefaults, *args, **kwargs
            )

            f_locals = dict(self.locals)
            f_locals.update(parsed_args)

            frame = Frame(
                code, self.builtins, self.globals, f_locals
            )  # Run code in prepared environment
            result = frame.run()
            return result

        self.push(f)

    # store

    def store_attr_op(self, name: str) -> None:
        obj = self.pop()
        value = self.pop()
        setattr(obj, name, value)

    def store_fast_op(self, name: str) -> None:
        self.locals[name] = self.pop()

    def store_slice_op(self, _: Any) -> None:
        a = self.pop()
        b = self.pop()
        c = self.pop()
        d = self.pop()
        c[b:a] = d
        self.push(c)

    def store_subscr_op(self, _: Any) -> None:
        key = self.pop()
        container = self.pop()
        value = self.pop()
        container[key] = value
        self.push(container)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.12.5/library/dis.html#opcode-STORE_NAME
        """
        const = self.pop()
        self.locals[arg] = const

    def store_global_op(self, arg: str) -> None:
        const = self.pop()
        self.globals[arg] = const

    # delete

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        delattr(obj, name)

    def delete_name_op(self, arg: str) -> None:
        del self.locals[arg]

    def delete_global_op(self, arg: str) -> None:
        del self.globals[arg]

    # import

    def import_name_op(self, namei: str) -> None:
        fromlist = self.pop()
        level = self.pop()
        self.push(__import__(namei, fromlist, level))

    def import_from_op(self, namei: str) -> None:
        module = self.data_stack[-1]
        self.push(getattr(module, namei))

    # operations

    def swap_op(self, i: int) -> None:
        stack = self.data_stack
        if len(stack) >= i:
            stack[-i], stack[-1] = stack[-1], stack[-i]

    def unary_negative_op(self, _: Any) -> None:
        self.push(-self.pop())

    def unary_invert_op(self, _: Any) -> None:
        self.push(~self.pop())

    def unary_not_op(self, _: Any) -> None:
        self.push(not self.pop())

    def binary_op_op(self, arg: int) -> None:
        ops = {
            0: lambda a, b: a + b,
            1: lambda a, b: a & b,
            2: lambda a, b: a // b,
            3: lambda a, b: a << b,
            4: lambda a, b: a @ b,
            5: lambda a, b: a * b,
            6: lambda a, b: a % b,
            7: lambda a, b: a | b,
            8: lambda a, b: a**b,
            9: lambda a, b: a >> b,
            10: lambda a, b: a - b,
            11: lambda a, b: a / b,
            12: lambda a, b: a ^ b,
        }
        a, b = self.pop(), self.pop()
        self.push(ops[arg % 13](b, a))

    def compare_op_op(self, arg: str) -> None:
        result = {
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }
        a, b = self.pop(), self.pop()
        self.push(result[arg](b, a))

    def contains_op_op(self, arg: int) -> None:
        a, b = self.pop(), self.pop()
        if arg:
            self.push(b not in a)
        else:
            self.push(b in a)

    def is_op_op(self, arg: int) -> None:
        a, b = self.pop(), self.pop()
        if arg:
            self.push(b is not a)
        else:
            self.push(b is a)

    # loop

    def get_iter_op(self, _: Any) -> None:
        self.push(iter(self.pop()))

    # for

    def for_iter_op(self, delta: int) -> None:
        iterator = self.pop()
        try:
            next_item = next(iterator)
            self.push(iterator)
            self.push(next_item)
        except StopIteration:
            self.instruction_num = delta

    def end_for_op(self, _: Any) -> None:
        pass

    # slices

    def binary_slice_op(self, _: Any) -> None:
        a = self.pop()
        b = self.pop()
        s = self.pop()
        self.push(s[b:a])

    def build_slice_op(self, arg: int) -> None:
        args = self.popn(arg)
        self.push(slice(*args))

    def binary_subscr_op(self, _: Any) -> None:
        a = self.pop()
        b = self.pop()
        self.push(b[a])

    def delete_subscr_op(self, _: Any) -> None:
        slice = self.pop()
        seq = self.pop()
        del seq[slice]
        self.push(seq)

    # tuple

    def build_tuple_op(self, arg: int) -> None:
        self.push(tuple(self.popn(arg)))

    # list

    def list_append_op(self, arg: int) -> None:
        self.data_stack[-arg - 1].append(self.pop())

    def build_list_op(self, arg: int) -> None:
        self.push(self.popn(arg))

    def list_extend_op(self, _: Any) -> None:
        seq = self.pop()
        self.data_stack[-1].extend(seq)

    # string

    def build_string_op(self, count: int) -> None:
        self.push("".join(self.popn(count)))

    # dict

    def build_map_op(self, arg: int) -> None:
        result = {}
        for _ in range(arg):
            value = self.pop()
            key = self.pop()
            result[key] = value
        self.push(result)

    def build_const_key_map_op(self, arg: int) -> None:
        keys = self.pop()
        values = self.popn(arg)
        self.push({key: value for key, value in zip(keys, values)})

    def map_add_op(self, i: int) -> None:
        value = self.pop()
        key = self.pop()
        self.data_stack[-i][key] = value

    def dict_merge_op(self, _: Any) -> None:
        seq = self.pop()
        self.data_stack[-1].update(seq)

    def dict_update_op(self, i: int) -> None:
        seq = self.pop()
        dict.update(self.data_stack[-i], seq)

    # set

    def build_set_op(self, arg: int) -> None:
        self.push(set(self.pop() for _ in range(arg)))

    def set_update_op(self, _: Any) -> None:
        seq = self.pop()
        self.data_stack[-1].update(seq)

    def set_add_op(self, i: int) -> None:
        elem = self.pop()
        set.add(self.data_stack[-i], elem)

    # exceptions

    def push_exc_info_op(self, _: Any) -> None:
        value = self.pop()
        self.push(None)
        self.push(value)

    def load_assertion_error_op(self, _: Any) -> None:
        self.push(AssertionError())

    def raise_varargs_op(self, arg: int) -> None:
        if arg == 0:
            raise
        elif arg == 1:
            raise self.pop()
        elif arg == 2:
            a, b = self.pop(), self.pop()
            raise b from a

    # annotations

    def setup_annotations_op(self, _: Any) -> None:
        if "__annotations__" not in self.locals:
            self.locals["__annotations__"] = {}

    # class

    def load_build_class_op(self, _: Any) -> None:
        self.push(builtins.__build_class__)


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(
            code_obj,
            builtins.globals()["__builtins__"],
            globals_context,
            globals_context,
        )
        return frame.run()
