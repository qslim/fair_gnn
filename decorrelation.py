import math
import torch
import torch.nn.functional as F


def orthogonal_projection(output, output_sens, config):
    output = output.squeeze()
    output_mean = output.mean()
    output_sens = output_sens.squeeze()
    output_sens_c = output_sens - output_sens.mean()
    output = ((output - output_mean) - config['orthogonality'] * ((output - output_mean) * output_sens_c).sum() / (
                output_sens_c.pow(2).sum() + 1e-8) * output_sens_c + output_mean).unsqueeze(-1)
    return output


def pow_scale_decorrelation(output, output_sens, config):
    # y_score, s_score = torch.sigmoid(output), torch.sigmoid(output_sens)
    # ms_cor = torch.abs(torch.mean((output_sens - torch.mean(output_sens)) * (output - torch.mean(output))))
    ms_cor = 0.0
    output, output_sens = output.squeeze(), output_sens.squeeze()
    for p in config['ms_bank']:
        if config['dataset'] == 'income' or config['dataset'] == 'bail':
            _output, _output_sens = output.abs().pow(p) * (output / output.abs()), output_sens.abs().pow(p) * (output_sens / output_sens.abs())
        else:
            _output, _output_sens = output.abs().pow(p), output_sens.abs().pow(p)
        # _output, _output_sens =output.pow(p), output_sens.pow(p)

        # ms_cor = ms_cor + torch.abs(torch.mean((_output_sens - torch.mean(_output_sens)) * (_output - torch.mean(_output))))

        _output, _output_sens = _output.unsqueeze(0), _output_sens.unsqueeze(0)
        _output, _output_sens = _output - torch.mean(_output), _output_sens - torch.mean(_output_sens)
        ms_cor = ms_cor + F.cosine_similarity(_output_sens, _output).abs().squeeze()

    return ms_cor


# def covariance(output, output_sens, config):
#     cov = torch.cat((output, output_sens), dim=1).transpose(0, 1)
#     return (cov[0][1] + cov[1][0]) / (cov[0][0] + cov[1][1])


# def moment(output, output_sens, config):
#     output, output_sens = output.squeeze(0), output_sens.squeeze(0)
#     output_c, output_sens_c = output - torch.mean(output), output_sens - torch.mean(output_sens)
#     ms_moment = 0.0
#     for i in [1, 2, 3]:
#         output_c, output_sens_c = output_c.pow(i), output_sens_c.pow(i)
#         ms_moment = ms_moment + ((output_c * output_sens_c).mean() - (output_c.mean() * output_sens_c.mean())).abs()
#     return ms_moment


# def pow_scale_decorrelation2(output, output_sens, config):
#     output, output_sens = output.squeeze(), output_sens.squeeze()
#     output_mat, output_sens_mat = [], []
#     for p in config['ms_bank']:
#         _output, _output_sens = output.abs().pow(p) * (output / output.abs()), output_sens.abs().pow(p) * (output_sens / output_sens.abs())
#         _output, _output_sens = _output.unsqueeze(0), _output_sens.unsqueeze(0)
#         output_mat.append(_output)
#         output_sens_mat.append(_output_sens)
#     output_mat, output_sens_mat = torch.cat(output_mat, dim=0), torch.cat(output_sens_mat, dim=0)
#     output_mat, output_sens_mat = output_mat - torch.mean(output_mat, dim=1, keepdim=True), output_sens_mat - torch.mean(output_sens_mat, dim=1, keepdim=True)
#     # ms_cor = F.cosine_similarity(output_sens_mat.unsqueeze(1), output_mat.unsqueeze(0)).abs().sum()
#     ms_cor = F.cosine_similarity(output_sens_mat, output_mat).abs().sum()
#     return ms_cor


def sin_scale_decorrelation(output, output_sens, config):
    # output, output_sens = F.sigmoid(output).squeeze(), F.sigmoid(output_sens).squeeze()
    output, output_sens = output.squeeze(), output_sens.squeeze()

    def encoding(e):
        # div = torch.exp(torch.arange(0, 32, 2) * (-math.log(100) / 32)).to(e.device) for bail
        div = torch.exp(torch.arange(0, 16, 2) * (-math.log(100) / 16)).to(e.device)
        pe = (e * 1.0).unsqueeze(1) * div
        return torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

    output_mat, output_sens_mat = encoding(output).transpose(0, 1), encoding(output_sens).transpose(0, 1)
    output_mat, output_sens_mat = output_mat - torch.mean(output_mat, dim=1, keepdim=True), output_sens_mat - torch.mean(output_sens_mat, dim=1, keepdim=True)
    ms_cor = F.cosine_similarity(output_sens_mat, output_mat).abs().sum()
    return ms_cor