import cntk
a = [1, 2, 3]
b = [4, 5, 6]
c = cntk.minus(a, b).eval()
print(c)