# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    V1 = int(data[0])
    V2 = int(data[1])
    V3 = int(data[2])

    if V1 + V2 + V3 != 3 * 7**3:
        print("No")
        return

    if V1 < 0 or V2 < 0 or V3 < 0:
        print("No")
        return

    if V1 > 7**3 or V2 > 7**3 or V3 > 7**3:
        print("No")
        return

    if V1 == 0 and V2 == 0 and V3 == 0:
        print("No")
        return

    if V1 == 7**3 and V2 == 0 and V3 == 0:
        print("No")
        return

    if V1 == 0 and V2 == 7**3 and V3 == 0:
        print("No")
        return

    if V1 == 0 and V2 == 0 and V3 == 7**3:
        print("No")
        return

    if V1 == 7**3 and V2 == 7**3 and V3 == 0:
        print("No")
        return

    if V1 == 7**3 and V2 == 0 and V3 == 7**3:
        print("No")
        return

    if V1 == 0 and V2 == 7**3 and V3 == 7**3:
        print("No")
        return

    if V1 == 7**3 and V2 == 7**3 and V3 == 7**3:
        print("No")
        return

    print("Yes")
    print("0 0 0 0 6 0 6 0 0")

if __name__ == "__main__":
    main()