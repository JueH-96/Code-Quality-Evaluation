import sys


def is_colorful_bracket_sequence(s: str) -> bool:
    """
    Returns True if the given bracket string `s` can be reduced to an empty
    string by repeatedly deleting the substrings (), [], <> (i.e.\ if it is a
    well-formed bracket sequence with the three types of brackets).
    """
    # Mapping from closing bracket to its corresponding opening bracket
    pair = {')': '(', ']': '[', '>': '<'}
    opening = set(pair.values())

    stack = []
    for ch in s:
        if ch in opening:            # opening bracket → push
            stack.append(ch)
        else:                        # closing bracket
            if not stack or stack[-1] != pair.get(ch):
                return False         # mismatch or empty stack
            stack.pop()              # matched → pop
    return not stack                  # must be empty if all matched


def main() -> None:
    s = sys.stdin.readline().strip()
    print("Yes" if is_colorful_bracket_sequence(s) else "No")


if __name__ == "__main__":
    main()