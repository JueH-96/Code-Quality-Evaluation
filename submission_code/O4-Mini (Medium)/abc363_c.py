def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1]); S = data[2].strip()
    # Count of each letter
    cnt = [0]*26
    for ch in S:
        cnt[ord(ch)-97] += 1
    # Unique letter indices to iterate
    uniq = [i for i in range(26) if cnt[i] > 0]
    path = [0]*N
    total = 0
    half = K // 2

    def dfs(pos):
        nonlocal total
        if pos == N:
            total += 1
            return
        # Try each available letter
        for i in uniq:
            if cnt[i] == 0:
                continue
            # place letter i at position pos
            path[pos] = i
            cnt[i] -= 1
            # If we have at least K characters, check the last K for palindrome
            if pos+1 >= K:
                start = pos - K + 1
                # check palindrome of path[start ... pos]
                pal = True
                # compare half pairs
                for j in range(half):
                    if path[start + j] != path[pos - j]:
                        pal = False
                        break
                if pal:
                    # prune: it forms a palindrome substring
                    cnt[i] += 1
                    continue
            # recurse
            dfs(pos+1)
            # backtrack
            cnt[i] += 1

    dfs(0)
    print(total)

if __name__ == "__main__":
    main()