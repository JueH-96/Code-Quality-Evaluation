import sys

def main() -> None:
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    N = int(data[0])
    prev = None  # previous dish taste
    for i in range(1, N + 1):  # dishes are in data[1:] 
        taste = data[i].strip()
        # Check if current and previous are both sweet
        if taste == "sweet" and prev == "sweet":
            # Takahashi becomes sick here (after eating this dish).
            # If this is not the last dish, he cannot finish all dishes.
            if i != N:
                print("No")
                return
            else:
                # Sick on the last dish – he has eaten all dishes.
                print("Yes")
                return
        prev = taste
    # Went through all dishes without a blocking sickness
    print("Yes")

if __name__ == "__main__":
    main()