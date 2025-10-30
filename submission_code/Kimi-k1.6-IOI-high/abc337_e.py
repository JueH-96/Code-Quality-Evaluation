import sys

def main():
    N = int(sys.stdin.readline())
    M = (N - 1).bit_length()
    
    friends = []
    for i in range(M):
        mask = 1 << i
        lst = [x for x in range(1, N + 1) if (x - 1) & mask]
        friends.append(lst)
    
    print(M)
    for lst in friends:
        print(len(lst), ' '.join(map(str, lst)))
    
    S = sys.stdin.readline().strip()
    s_rev = S[::-1]
    x = int(s_rev, 2) + 1
    print(x)

if __name__ == "__main__":
    main()