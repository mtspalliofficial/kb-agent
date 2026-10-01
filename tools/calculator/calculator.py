TOOL_INFO = {
    "name": "calculator",
    "description": "Evaluates basic arithmetic expressions (+, -, *, /, parentheses, decimals)."
}


def run(expression):
    """
    Safely evaluates a basic arithmetic expression.

    Allowed:
    - Numbers
    - +
    - -
    - *
    - /
    - %
    - Parentheses
    - Decimal points
    - Spaces
    """

    allowed_chars = "0123456789+-*/().% "

    if not expression or not expression.strip():
        return "Error: no expression provided."

    expression = expression.strip()

    # Make sure only allowed characters are present.
    if not all(char in allowed_chars for char in expression):
        return "Error: expression contains unsupported characters."

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except ZeroDivisionError:
        return "Error: division by zero."

    except SyntaxError:
        return "Error: invalid expression syntax."

    except Exception as e:
        return f"Error: could not evaluate expression ({e})"


if __name__ == "__main__":

    print(run("133 * 165"))
    print(run("15/100 * 2000"))
    print(run("25 * 800"))
    print(run("(150 + 50) / 4"))
    print(run("200 * 0.05 * 3"))
    print(run("10 / 0"))
    print(run("import os"))