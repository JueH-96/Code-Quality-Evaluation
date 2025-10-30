import sys

def solve():
    S = sys.stdin.readline().strip()

    # The problem asks to check characters at "every even number i from 2 through 16".
    # This implies 1-based indexing for 'i'.
    # In 0-based Python indexing, these correspond to:
    # 2nd character (i=2) -> S[1]
    # 4th character (i=4) -> S[3]
    # ...
    # 16th character (i=16) -> S[15]
    # So we need to check characters at 0-indexed positions 1, 3, 5, 7, 9, 11, 13, 15.

    # We can use a loop or the all() function for a concise check.
    # The all() function returns True if all elements in an iterable are true.
    # Here, we create a generator expression that yields True if S[i] is '0', else False.
    # If all these checks yield True, then all relevant characters are '0'.
    
    # Iterate through the required 0-indexed positions: 1, 3, ..., 15
    # range(start, stop, step) generates numbers from 'start' up to (but not including) 'stop', with 'step'.
    # range(1, 16, 2) will generate: 1, 3, 5, 7, 9, 11, 13, 15
    
    is_all_even_positioned_chars_zero = all(S[i] == '0' for i in range(1, 16, 2))

    if is_all_even_positioned_chars_zero:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()