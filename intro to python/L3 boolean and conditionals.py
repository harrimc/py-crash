def onionless(ketchup, mustard, onion):
    """Return whether the customer doesn't want onions.
    """
    return not onion

def noketchup(ketchup ,mustard, onion) :
    ''' takes arguement of ketchup and assumes you want none'''
    return not ketchup 

def nomust(ketchup, mustard, onion) :
    '''asumes you dont want mustard'''
    return not mustard

print('would you like onions?')
ask = input('> ').lower()
onion  = (ask == 'no')
reton = onionless(0,0,onion)

print('Would you like ketchup?')
ket = input('> ').lower()
ketty = (ket == 'no')
retket = noketchup(ketty,0,0)

print('Would you like mustard?')
mus = input('> ').lower()
mussy = (mus == 'no')
retmut = nomust(0,mussy,0)

print('On your burger you want')
if reton == True :
    print('Onions')
if retket == True :
    print('Ketchup')
if retmut == True :
    print('Mustard')