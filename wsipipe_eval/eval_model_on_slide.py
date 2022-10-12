import numpy as np
import pandas as pd
import torch


def eval_model_on_slide(model, slide_loader, device, classes_out):

    model.eval()
    
    num_samples = len(slide_loader) * slide_loader.batch_size

    prob_out = np.zeros((num_samples, len(classes_out)))

    with torch.no_grad():
        for idx, batch in enumerate(slide_loader):
            data, target = batch
            data = data.to(device)
            output = model(data)
            sm = torch.nn.Softmax(1)
            output_sm = sm(output)
            pred_prob = output_sm.cpu().numpy()  # rows: batch_size, cols: num_classes
            start = idx * slide_loader.batch_size
            end = start + pred_prob.shape[0]
            prob_out[start:end, :] = pred_prob

            if len(slide_loader) >= 500:
                if idx % 100 == 0:
                    print('Batch {} of {} on GPU {}'.format(idx, len(slide_loader), device))
            elif len(slide_loader) >= 200:
                if idx % 50 == 0:
                    print('Batch {} of {} on GPU {}'.format(idx, len(slide_loader), device))
            elif len(slide_loader) >= 100:
                if idx % 25 == 0:
                    print('Batch {} of {} on GPU {}'.format(idx, len(slide_loader), device))
            else:
                if idx % 10 == 0:
                    print('Batch {} of {} on GPU {}'.format(idx, len(slide_loader), device))

    # remove blank padding for last batch
    prob_out = prob_out[0:slide_loader.dataset.patch_df.shape[0], :]
    prob_out = pd.DataFrame(prob_out, columns=sorted(classes_out))
    slide_df = pd.concat((slide_loader.dataset.patch_df, prob_out), axis=1)

    return slide_df
