import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); k = int(data[idx+1]); idx += 2
        s = data[idx]; idx += 1

        # Count how many characters have odd frequency
        freq = [0]*26
        for ch in s:
            freq[ord(ch) - ord('a')] += 1
        odd = 0
        for f in freq:
            if f & 1:
                odd += 1

        # After deleting k chars, the remaining length is n-k.
        # A palindrome of length L can have at most (L mod 2) odd‐count chars.
        allowed = (n - k) & 1

        # We must delete at least (odd - allowed) odd‐parity chars (each deletion
        # from an odd‐count char reduces the number of odd parities by 1).
        # If odd <= allowed, no mandatory deletions needed.
        need = odd - allowed
        if need < 0:
            need = 0

        if k >= need:
            out.append("YES")
        else:
            out.append("NO")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()