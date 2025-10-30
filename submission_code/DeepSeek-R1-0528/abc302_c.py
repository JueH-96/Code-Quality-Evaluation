import itertools

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    
    n, m = map(int, data[0].split())
    words = []
    for i in range(1, 1+n):
        words.append(data[i].strip())
    
    for perm in itertools.permutations(words):
        valid_perm = True
        for i in range(len(perm)-1):
            diff_count = 0
            for j in range(m):
                if perm[i][j] != perm[i+1][j]:
                    diff_count += 1
                    if diff_count > 1:
                        break
            if diff_count != 1:
                valid_perm = False
                break
        if valid_perm:
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    main()