A, B = map(int, input().split())

def get_position(num):
    row = (num - 1) // 3
    col = (num - 1) % 3
    return (row, col)

pos_A = get_position(A)
pos_B = get_position(B)

if pos_A[0] == pos_B[0] and abs(pos_A[1] - pos_B[1]) == 1:
    print("Yes")
else:
    print("No")