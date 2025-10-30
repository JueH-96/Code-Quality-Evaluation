import sys

def main():
    S, T = input().split()
    len_S = len(S)
    
    for w in range(1, len_S):
        for c in range(1, w + 1):
            collected = []
            start = 0
            while start < len_S:
                end = start + w
                sub = S[start:end]
                if len(sub) >= c:
                    collected.append(sub[c-1])
                start = end
            candidate = ''.join(collected)
            if candidate == T:
                print("Yes")
                return
    print("No")

if __name__ == "__main__":
    main()