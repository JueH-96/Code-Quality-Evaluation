def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    index = 1
    
    answers = []
    for _ in range(t):
        n = int(data[index]); index += 1
        arr = list(map(int, data[index:index+n]))
        index += n
        
        current_sum = arr[0]
        max_sum = arr[0]
        
        for i in range(1, n):
            # Check if current element has different parity than the previous one
            if (arr[i] % 2) != (arr[i - 1] % 2):
                current_sum += arr[i]
            else:
                current_sum = arr[i]
            max_sum = max(max_sum, current_sum)
        
        answers.append(str(max_sum))
    
    print("
".join(answers))

# Do not forget to call main()
if __name__ == '__main__':
    main()