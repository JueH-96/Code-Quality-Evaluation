import sys

def is_321_like(num_str: str) -> bool:
    """
    Returns True if the digits in num_str are strictly decreasing from left to right.
    """
    for i in range(len(num_str) - 1):
        if num_str[i] <= num_str[i + 1]:
            return False
    return True

def main() -> None:
    N = sys.stdin.readline().strip()           # read input as string to access digits directly
    print("Yes" if is_321_like(N) else "No")

if __name__ == "__main__":
    main()