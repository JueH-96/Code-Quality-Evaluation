def main():
    Q = int(input().strip())
    stack = [0] * 100  # Initial stack with 100 zeros
    
    for _ in range(Q):
        data = input().split()
        if data[0] == '1':
            x = int(data[1])
            stack.append(x)
        else:  # data[0] == '2'
            popped_value = stack.pop()
            print(popped_value)

if __name__ == "__main__":
    main()