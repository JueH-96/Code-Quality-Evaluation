m, d = map(int, input().split())
y, mo, da = map(int, input().split())
mo_ = mo * d + da
if mo_ + 1 > m * d:
    y += 1
    mo = 1
    da = 1
else:
    mo_ += 1
    mo = mo_ // d if mo_ % d == 0 else mo_ // d + 1
    da = d if mo_ % d == 0 else mo_ % d
print(y, mo, da)