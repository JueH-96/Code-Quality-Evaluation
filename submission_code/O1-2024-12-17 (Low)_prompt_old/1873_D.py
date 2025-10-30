def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1
    
    answers = []
    for _ in range(t):
        n, k = map(int, input_data[idx:idx+2])
        idx += 2
        s = input_data[idx]
        idx += 1
        
        last_covered = -1
        operations = 0
        for i in range(n):
            if s[i] == 'B' and i > last_covered:
                operations += 1
                last_covered = i + k - 1
        
        answers.append(str(operations))
    
    print("
".join(answers))

def main():
    solve()

if __name__ == "__main__":
    main()