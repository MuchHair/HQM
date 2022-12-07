
def build_model(args):

    if args.model_name=='CDN_DN':
        from models.DN.hoi_doq import build
    elif args.model_name=='CDN_HQM':
        from models.DN.hoi_hqm import build
    elif args.model_name=='CDN_HQM_W':
        from models.DN.hoi_hqm_w import build
    else:
        assert False
        from .hoi import build

    return build(args)
