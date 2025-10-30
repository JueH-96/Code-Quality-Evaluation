import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        freq = {}
        for i in range(n):
            a = int(data[idx]); idx += 1
            freq[a] = freq.get(a, 0) + 1
        # Count pairs with equal a_i
        total = 0
        for v in freq.values():
            if v > 1:
                total += v * (v - 1) // 2
        # Add special cross pairs for (1,2)
        total += freq.get(1, 0) * freq.get(2, 0)
        out.append(str(total))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()