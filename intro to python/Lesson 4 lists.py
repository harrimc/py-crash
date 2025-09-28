
party_attendees = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']

def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    if name == party_attendees[-1] :
        print(name, ' was unfasionably late!')
    elif party_attendees.index(name) > arrivals/2 :
        print(name, ' was fasionably late')
    else :
        print(name, ' was not fasionably late')      



print('Who would you like to check?')
na = input('> ')
fashionably_late(len(party_attendees), na)
