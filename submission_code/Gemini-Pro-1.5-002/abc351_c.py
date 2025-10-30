# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    seq = []
    
    for x in a:
        seq.append(x)
        while len(seq) > 1:
            if seq[-1] == seq[-2]:
                seq.pop()
                seq.pop()
                seq.append(seq[-1] + 1 if len(seq) > 0 else x + 1)
            else:
                break
    
    print(len(seq))

solve()