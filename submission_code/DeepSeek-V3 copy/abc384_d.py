# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    total_sum = sum(A)
    
    if S == 0:
        print("Yes")
        return
    
    if total_sum == 0:
        if S == 0:
            print("Yes")
        else:
            print("No")
        return
    
    if S > total_sum * 2:
        print("No")
        return
    
    prefix = [0]
    for num in A:
        prefix.append(prefix[-1] + num)
    
    for num in A:
        prefix.append(prefix[-1] + num)
    
    target = S
    seen = {0}
    current_sum = 0
    
    for i in range(len(prefix)):
        current_sum = prefix[i]
        if (current_sum - target) in seen:
            print("Yes")
            return
        seen.add(current_sum)
    
    print("No")

if __name__ == "__main__":
    main()