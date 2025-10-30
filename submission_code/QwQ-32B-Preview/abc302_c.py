import itertools

def diff_by_one(s1, s2):
    count = 0
    for a, b in zip(s1, s2):
        if a != b:
            count += 1
            if count > 1:
                return False
    return count == 1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    strings = data[2:2+N]
    
    for perm in itertools.permutations(strings):
        valid = True
        for i in range(N-1):
            if not diff_by_one(perm[i], perm[i+1]):
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()