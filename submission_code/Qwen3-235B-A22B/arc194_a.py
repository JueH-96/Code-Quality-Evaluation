def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    stack = []
    current_sum = 0
    best = float('-inf')
    
    for a in A:
        if not stack:
            stack.append(a)
            current_sum = a
        else:
            option_append = current_sum + a
            option_delete = current_sum - stack[-1]
            if option_append > option_delete:
                stack.append(a)
                current_sum = option_append
            else:
                stack.pop()
                current_sum = option_delete
        if current_sum > best:
            best = current_sum
    print(best)

if __name__ == "__main__":
    main()