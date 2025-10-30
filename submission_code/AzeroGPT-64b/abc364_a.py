import sys

N = int(input())
dishes = [input() for _ in range(N)]

prev_dish = "none"
for dish in dishes:
    if dish == "sweet" and prev_dish == "sweet":
        print("No")
        sys.exit()
    prev_dish = dish
print("Yes")