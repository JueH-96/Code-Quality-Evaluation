def main():
    import sys
    data = sys.stdin.read().split()
    sx, sy, tx, ty = map(int, data)
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    # When there is no vertical move, the horizontal toll is paid
    # only when we cross a seam between different tiles.
    # That's floor((dx + parity) / 2), where parity = (sx + sy) % 2.
    if dy == 0:
        parity = (sx + sy) & 1
        horiz = (dx + parity) // 2
        print(horiz)
    else:
        # If we have at least one vertical move, each vertical step
        # lets us shift row parity and thus avoid one horizontal toll
        # per vertical move (and one more at the start),
        # so we can make up to (dy+1) horizontal crossings toll‐free.
        # Remaining horizontal crossings each cost 1.
        free = dy + 1
        paid_horiz = dx - free
        if paid_horiz < 0:
            paid_horiz = 0
        # plus every vertical move costs 1
        print(paid_horiz + dy)

if __name__ == "__main__":
    main()