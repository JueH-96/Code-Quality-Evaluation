import sys

def main():
    data = sys.stdin.readline().strip()
    N = int(data)
    
    M = 0
    while (1 << M) < N:
        M += 1
    
    print(M)
    sys.stdout.flush()
    
    if M == 0:
        s = sys.stdin.readline().strip()
        print(1)
        sys.stdout.flush()
        return

    lists = [[] for _ in range(M)]
    
    for j in range(1, N+1):
        num_val = j - 1
        for i in range(M):
            if num_val & (1 << i):
                lists[i].append(j)
    
    for i in range(M):
        k = len(lists[i])
        if k == 0:
            print("0")
        else:
            line = str(k) + " " + " ".join(map(str, lists[i]))
            print(line)
        sys.stdout.flush()
    
    s = sys.stdin.readline().strip()
    num = 0
    for i in range(len(s)):
        if s[i] == '1':
            num += (1 << i)
            
    result = num + 1
    print(result)
    sys.stdout.flush()

if __name__ == "__main__":
    main()