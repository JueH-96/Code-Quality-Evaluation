import sys

def main():
    data = sys.stdin
    first = data.readline().split()
    if not first:
        return
    N = int(first[0])
    t = first[1]
    nt = len(t)
    res = []
    for idx in range(1, N+1):
        s = data.readline().rstrip('
')
        ns = len(s)
        d = nt - ns
        if d == 0:
            # allow zero or one substitution
            if s == t:
                res.append(idx)
            else:
                mismatches = 0
                for i in range(ns):
                    if s[i] != t[i]:
                        mismatches += 1
                        if mismatches == 2:
                            break
                if mismatches == 1:
                    res.append(idx)
        elif d == 1:
            # t is longer by one: check if deleting one char from t gives s
            i = j = 0
            while i < ns and j < nt and s[i] == t[j]:
                i += 1
                j += 1
            j += 1  # skip the extra char in t
            while i < ns and j < nt and s[i] == t[j]:
                i += 1
                j += 1
            if i == ns and j == nt:
                res.append(idx)
        elif d == -1:
            # s is longer by one: check if deleting one char from s gives t
            i = j = 0
            while i < ns and j < nt and s[i] == t[j]:
                i += 1
                j += 1
            i += 1  # skip the extra char in s
            while i < ns and j < nt and s[i] == t[j]:
                i += 1
                j += 1
            if i == ns and j == nt:
                res.append(idx)
        # else |d| > 1, impossible; skip

    out = sys.stdout
    out.write(str(len(res)) + "
")
    if res:
        out.write(" ".join(map(str, res)) + "
")

if __name__ == "__main__":
    main()