def remove_parentheses(s):
    """
    Removes all possible substrings that start with '(' and end with ')'
    without any '(' or ')' in between.

    Args:
        s (str): The input string.

    Returns:
        str: The string after removing all possible substrings.
    """
    stack = []
    result = []
    for char in s:
        if char == '(':
            stack.append(len(result))
        elif char == ')':
            if stack:
                start = stack.pop()
                result = result[:start] + result[start + 1:]
        else:
            result.append(char)
    return ''.join(result)


if __name__ == "__main__":
    n = int(input())
    s = input()
    print(remove_parentheses(s))