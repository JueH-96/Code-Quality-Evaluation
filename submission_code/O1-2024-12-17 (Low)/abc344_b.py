def main():
    values = []
    # Read lines until a 0 is encountered
    while True:
        num_str = input().strip()
        num = int(num_str)
        values.append(num)
        if num == 0:
            break
    
    # Print in reverse order
    for v in reversed(values):
        print(v)

# Do not forget to call main
main()