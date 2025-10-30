from math import pow

def answer() :
    max_num = int(input())

    sq = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 2)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/2)) + 1)
                    )
                )
            )
        )
    )
    th = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 3)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/3)) + 1)
                    )
                )
            )
        )
    )
    fo = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 4)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/4)) + 1)
                    )
                )
            )
        )
    )
    fi = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 5)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/5)) + 1)
                    )
                )
            )
        )
    )
    si = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 6)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/6)) + 1)
                    )
                )
            )
        )
    )
    se = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 7)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/7)) + 1)
                    )
                )
            )
        )
    )
    ei = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 8)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/8)) + 1)
                    )
                )
            )
        )
    )
    ni = list(
        set(
            list(
                map(
                    lambda x : int(pow(int(x), 9)),
                    # make iterable
                    map(
                        str, 
                        range(2, int(pow(max_num, 1/9)) + 1)
                    )
                )
            )
        )
    )

    print(
        len(
            set(
                sq + th + fo +  fi + si + se + ei + ni
            )
        )
    )

answer()