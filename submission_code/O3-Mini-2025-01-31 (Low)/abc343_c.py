def is_palindrome(num: int) -> bool:
    s = str(num)
    return s == s[::-1]

def main():
    import sys
    import math
    
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = int(input_data)
    
    # Find the maximum integer x such that x^3 <= N
    # Using math.floor to determine potential cube root.
    x_max = int(round(N ** (1/3)))
    # Adjust x_max if necessary: sometimes rounding errors happen.
    while (x_max + 1) ** 3 <= N:
        x_max += 1
    while x_max ** 3 > N and x_max > 0:
        x_max -= 1

    # Traverse backwards from x_max down to 1
    for x in range(x_max, 0, -1):
        cube = x ** 3
        # If palindrome and cube <= N, that's the maximum because we're descending.
        if is_palindrome(cube):
            sys.stdout.write(str(cube))
            return

if __name__ == "__main__":
    main()