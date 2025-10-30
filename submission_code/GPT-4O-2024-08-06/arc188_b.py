# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    T = int(data[0])
    index = 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        if N % 2 == 0 and K == N // 2:
            results.append("No")
        else:
            results.append("Yes")
    
    sys.stdout.write("
".join(results) + "
")