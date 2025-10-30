import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]

    unique_perms = set(itertools.permutations(S))
    count = 0

    for perm in unique_perms:
        s = ''.join(perm)
        flag = True
        for i in range(N - K + 1):
            substring = s[i:i+K]
            if substring == substring[::-1]:
                flag = False
                break
        if flag:
            count += 1

    print(count)

if __name__ == "__main__":
    main()