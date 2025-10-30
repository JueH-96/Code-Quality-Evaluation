def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    sequence = []
    
    for i in range(n):
        sequence.append(a[i])
        
        while len(sequence) >= 2:
            if sequence[-1] == sequence[-2]:
                sequence[-2] += 1
                sequence.pop()
            else:
                break
                
    print(len(sequence))

solve()