import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    out = []
    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        # If n is divisible by 3, Vanya (First) cannot win in his first move,
        # and Vova can always return the number to ≡0 mod 3 on his turns,
        # so Vova wins (Second). Otherwise Vanya wins immediately.
        if n % 3 == 0:
            out.append("Second")
        else:
            out.append("First")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()