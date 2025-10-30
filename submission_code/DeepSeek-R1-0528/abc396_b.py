def main():
    Q = int(input().strip())
    stack = [0] * 100
    ans = []
    
    for _ in range(Q):
        data = input().split()
        if data[0] == '1':
            x = int(data[1])
            stack.append(x)
        else:
            top_val = stack.pop()
            ans.append(top_val)
            
    for val in ans:
        print(val)

if __name__ == "__main__":
    main()