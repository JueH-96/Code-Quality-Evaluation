def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    n = int(input_data[0])
    dishes = input_data[1:]
    
    # Simulate eating each dish.
    # If Takahashi eats two sweet dishes consecutively and it's not the last dish,
    # he becomes sick and stops, so he won't eat all the dishes.
    # Note: If the consecutive sweets occur at the very end (last two dishes),
    # then he successfully consumes all dishes.
    for i in range(1, n):
        if dishes[i] == "sweet" and dishes[i - 1] == "sweet":
            if i != n - 1:  # If not the last dish, he stops early.
                print("No")
                return
    print("Yes")
    
if __name__ == "__main__":
    main()