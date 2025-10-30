import sys

def is_326_like(num: int) -> bool:
    """
    Return True if num is a 326-like number, False otherwise.
    """
    h = num // 100          # hundreds digit
    t = (num // 10) % 10    # tens digit
    o = num % 10            # ones digit
    return h * t == o

def main() -> None:
    N = int(sys.stdin.readline().strip())

    for candidate in range(N, 1000):          # 100 .. 999
        if is_326_like(candidate):
            print(candidate)
            return

if __name__ == "__main__":
    main()