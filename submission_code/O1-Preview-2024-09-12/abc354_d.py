# YOUR CODE HERE
def main():
    A,B,C,D=map(int,input().split())
    W = C - A
    H = D - B

    total_area = W * H

    color = ((A - B) % 2 + 2 ) % 2

    if total_area % 2 == 0:
        black_area = total_area // 2
    else:
        black_area = total_area //2 + color

    print(black_area * 2)

main()