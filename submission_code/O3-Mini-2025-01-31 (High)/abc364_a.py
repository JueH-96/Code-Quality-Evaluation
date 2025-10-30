def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    dishes = data[1:]
    
    # Process each dish in order.
    for i in range(1, n):
        if dishes[i] == "sweet" and dishes[i - 1] == "sweet":
            # Takahashi becomes sick after eating the dish i.
            # If dish i is not the last dish, he cannot continue.
            if i != n - 1:
                print("No")
                return
    print("Yes")

if __name__ == '__main__':
    main()