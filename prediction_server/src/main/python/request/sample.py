import requests


def main():
    # Single sample request coming from back-end
    sample_test_data = {'id': 0, 'gender': 1, 'age': 1, 'height': 11, 'weight': 1, '4030952': None, '6739157': None, '6828931': None,
                        '6834113': None, '6840157': None, '6842280': None, '6842456': None, '6843136': None, '6843348': None,
                        '6846168': None, '6847470': None, '6847884': None, '6850385': None, '6852493': 4, '6854799': None, '6854893': None,
                        '6857631': 3, '6862339': None, '6863696': None, '6865791': None, '6866122': None, '6867736': None, '6868389': None,
                        '6869017': None, '6869489': None, '6871953': None, '6872334': None, '6872474': None, '6875304': None, '6877253': None,
                        '6878340': None, '6881099': None, '6881146': None, '6881286': None, '6881815': None, '6882107': None, '6883649': None,
                        '6884169': None, '6885805': None, '6885909': 3, '6886357': None, '6886567': None, '6891606': None, '6891823': None,
                        '6892138': None, '6892209': None, '6892442': None, '6894323': None, '6894755': None, '6895112': None, '6895591': 3,
                        '6895723': None, '6896819': None, '6897374': None, '6897383': None, '6897704': None, '6898328': None, '6898497': None,
                        '6899265': None, '6900601': None, '6901939': None, '6902207': None, '6902300': None, '6902441': None, '6904080': None,
                        '6904344': None, '6904476': None, '6904643': None, '6905276': None, '6906436': None, '6907091': None, '6912270': None,
                        '6912734': 2, '6913952': None, '6915088': None, '6916408': 3, '6918383': None, '6918977': None, '6919305': None, '6922190': None,
                        '6922433': None, '6922700': None, '6923484': None, '6924155': None, '6924266': None, '6925791': None, '6925795': None, '6927971': None,
                        '6928209': None, '6929388': None, '6929691': 4, '6930302': None, '6930645': None, '6932059': None, '6932517': None, '6932677': None,
                        '6932924': None, '6933005': None, '6933144': None, '6933349': None}


    r = requests.post(
        'http://localhost:5000/predict',
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        json=sample_test_data
    )
    print(r.json())


if __name__ == '__main__':
    main()
