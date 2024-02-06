scheduler = None
is_group_datasets = ['coral', 'groupdro', 'irm']

def _echr_init(args, is_group_data):
    if is_group_data:
        from .data.echr import ECHRGroup
        dataset = ECHRGroup(args)
    else:
        from .data.echr import ECHR
        dataset = ECHR(args)
    return dataset


def getdata(args, is_group_data = False):
    dataset_name = args.dataset
    if dataset_name == 'echr': return _echr_init(args, is_group_data)
