def get_back(val) :
    mean = sum(val)/len(val)
    maxim = max(val)
    std = (((sum(val)**2)/len(val)) - (sum(val)/len(val))**2)**0.5
    return mean,maxim,std

liz = [2, 4, 4, 4, 5, 5, 7, 9]
me, ma, st = get_back(liz)
print(me,ma,st)