import torch
import torch.nn.functional as F


def cam_to_label(cam, img_box=None, cls_label=None, ignore_mid=False, cfg=None):
    b, c, h, w = cam.shape
    cam_value, _pseudo_label = cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    # _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    pseudo_label = torch.ones_like(_pseudo_label)

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return cam, pseudo_label


def cam_to_label2(cam, img_box=None, cls_label=None, ignore_mid=False, bkg_score=None):
    b, c, h, w = cam.shape
    cam_value, _pseudo_label = cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    # _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    _pseudo_label[cam_value <= bkg_score] = 0
    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= bkg_score] = 0
    pseudo_label = torch.ones_like(_pseudo_label)

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return cam, pseudo_label


def multi_scale_cam(model, inputs_A, inputs_B, scales):
    #cam_list = []
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True)   #_cam: torch.Size([8, 1, 16, 16])

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        
        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True)  #_cam, _,_

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


def multi_scale_cam2(model, inputs_A, inputs_B, scales):
    #cam_list = []
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        # _cam4 = model(inputs_A_cat, inputs_B_cat, cam_only=True)
        _cam2, _cam3, _cam4 = model(inputs_A_cat, inputs_B_cat, cam_only=True)   #_cam: torch.Size([8, 1, 16, 16])

        _cam2 = F.interpolate(_cam2, size=(h, w), mode='bilinear', align_corners=False)
        _cam2 = torch.max(_cam2[:b, ...], _cam2[b:, ...].flip(-1))

        _cam3 = F.interpolate(_cam3, size=(h, w), mode='bilinear', align_corners=False)
        _cam3 = torch.max(_cam3[:b, ...], _cam3[b:, ...].flip(-1))

        _cam4 = F.interpolate(_cam4, size=(h, w), mode='bilinear', align_corners=False)
        _cam4 = torch.max(_cam4[:b, ...], _cam4[b:, ...].flip(-1))

        # for s in scales:
        #     if s != 1.0:
        #         _inputs_A = F.interpolate(inputs_A, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
        #         inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
        #         _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
        #         inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)
        #
        #         _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True)  #_cam, _,_
        #
        #         _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        #         _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        #
        #         cam_list.append(F.relu(_cam))

        cam_list2 = [F.relu(_cam2)]
        cam_list3 = [F.relu(_cam3)]
        cam_list4 = [F.relu(_cam4)]
        cam2 = torch.sum(torch.stack(cam_list2, dim=0), dim=0)
        cam2 = cam2 + F.adaptive_max_pool2d(-cam2, (1, 1))
        cam2 /= F.adaptive_max_pool2d(cam2, (1, 1)) + 1e-5
        cam3 = torch.sum(torch.stack(cam_list3, dim=0), dim=0)
        cam3 = cam3 + F.adaptive_max_pool2d(-cam3, (1, 1))
        cam3 /= F.adaptive_max_pool2d(cam3, (1, 1)) + 1e-5
        cam4 = torch.sum(torch.stack(cam_list4, dim=0), dim=0)
        cam4 = cam4 + F.adaptive_max_pool2d(-cam4, (1, 1))
        cam4 /= F.adaptive_max_pool2d(cam4, (1, 1)) + 1e-5

        # _cam2 = _cam2.reshape(b, 1, 4, h // 4, 4, w // 4).permute(0, 1, 2, 4, 3, 5).reshape(b * 16, 1, h // 4, w // 4)
        # _cam3 = _cam3.reshape(b, 1, 2, h // 2, 2, w // 2).permute(0, 1, 2, 4, 3, 5).reshape(b * 4, 1, h // 2, w // 2)
        # cam_list2 = [F.relu(_cam2)]
        # cam_list3 = [F.relu(_cam3)]
        # cam_list4 = [F.relu(_cam4)]
        # cam2 = torch.sum(torch.stack(cam_list2, dim=0), dim=0)
        # cam2 = cam2 + F.adaptive_max_pool2d(-cam2, (1, 1))
        # cam2 /= F.adaptive_max_pool2d(cam2, (1, 1)) + 1e-5
        # cam3 = torch.sum(torch.stack(cam_list3, dim=0), dim=0)
        # cam3 = cam3 + F.adaptive_max_pool2d(-cam3, (1, 1))
        # cam3 /= F.adaptive_max_pool2d(cam3, (1, 1)) + 1e-5
        # cam4 = torch.sum(torch.stack(cam_list4, dim=0), dim=0)
        # cam4 = cam4 + F.adaptive_max_pool2d(-cam4, (1, 1))
        # cam4 /= F.adaptive_max_pool2d(cam4, (1, 1)) + 1e-5
        # cam2 = cam2.reshape(b, 1, 4, 4, h // 4, w // 4).permute(0, 1, 2, 4, 3, 5).reshape(b, 1, h, w)
        # cam3 = cam3.reshape(b, 1, 2, 2, h // 2, w // 2).permute(0, 1, 2, 4, 3, 5).reshape(b, 1, h, w)

    return cam2, cam3, cam4


def multi_scale_cam3(model, inputs_A, inputs_B, scales):
    # cam_list = []
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        # _cam4 = model(inputs_A_cat, inputs_B_cat, cam_only=True)
        _cam2, _cam3, _cam4 = model(inputs_A_cat, inputs_B_cat, cam_only=True)  # _cam: torch.Size([8, 1, 16, 16])

        _cam2 = F.interpolate(_cam2, size=(h, w), mode='bilinear', align_corners=False)
        _cam2 = torch.max(_cam2[:b, ...], _cam2[b:, ...].flip(-1))

        _cam3 = F.interpolate(_cam3, size=(h, w), mode='bilinear', align_corners=False)
        _cam3 = torch.max(_cam3[:b, ...], _cam3[b:, ...].flip(-1))

        # c4 = _cam4.clone()
        # c4 = torch.max(c4[:b, ...], c4[b:, ...].flip(-1))
        # c4 = c4 + F.adaptive_max_pool2d(-c4, (1, 1))
        # c4 /= F.adaptive_max_pool2d(c4, (1, 1)) + 1e-5

        _cam4 = F.interpolate(_cam4, size=(h, w), mode='bilinear', align_corners=False)
        _cam4 = torch.max(_cam4[:b, ...], _cam4[b:, ...].flip(-1))

        # for s in scales:
        #     if s != 1.0:
        #         _inputs_A = F.interpolate(inputs_A, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
        #         inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
        #         _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
        #         inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)
        #
        #         _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True)  #_cam, _,_
        #
        #         _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        #         _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        #
        #         cam_list.append(F.relu(_cam))

        # cam_list2 = [F.relu(_cam2)]
        # cam_list3 = [F.relu(_cam3)]
        # cam_list4 = [F.relu(_cam4)]
        # cam2 = torch.sum(torch.stack(cam_list2, dim=0), dim=0)
        # cam2 = cam2 + F.adaptive_max_pool2d(-cam2, (1, 1))
        # cam2 /= F.adaptive_max_pool2d(cam2, (1, 1)) + 1e-5
        # cam3 = torch.sum(torch.stack(cam_list3, dim=0), dim=0)
        # cam3 = cam3 + F.adaptive_max_pool2d(-cam3, (1, 1))
        # cam3 /= F.adaptive_max_pool2d(cam3, (1, 1)) + 1e-5
        # cam4 = torch.sum(torch.stack(cam_list4, dim=0), dim=0)
        # cam4 = cam4 + F.adaptive_max_pool2d(-cam4, (1, 1))
        # cam4 /= F.adaptive_max_pool2d(cam4, (1, 1)) + 1e-5

        s2 = 1
        s3 = 1
        _cam2 = _cam2.reshape(b, 1, s2, h // s2, s2, w // s2).permute(0, 1, 2, 4, 3, 5).reshape(b * s2 * s2, 1, h // s2,
                                                                                                w // s2)
        _cam3 = _cam3.reshape(b, 1, s3, h // s3, s3, w // s3).permute(0, 1, 2, 4, 3, 5).reshape(b * s3 * s3, 1, h // s3,
                                                                                                w // s3)
        cam_list2 = [F.relu(_cam2)]
        cam_list3 = [F.relu(_cam3)]
        cam_list4 = [F.relu(_cam4)]
        cam2 = torch.sum(torch.stack(cam_list2, dim=0), dim=0)
        cam2 = cam2 + F.adaptive_max_pool2d(-cam2, (1, 1))
        cam2 /= F.adaptive_max_pool2d(cam2, (1, 1)) + 1e-5
        cam3 = torch.sum(torch.stack(cam_list3, dim=0), dim=0)
        cam3 = cam3 + F.adaptive_max_pool2d(-cam3, (1, 1))
        cam3 /= F.adaptive_max_pool2d(cam3, (1, 1)) + 1e-5
        cam4 = torch.sum(torch.stack(cam_list4, dim=0), dim=0)
        cam4 = cam4 + F.adaptive_max_pool2d(-cam4, (1, 1))
        cam4 /= F.adaptive_max_pool2d(cam4, (1, 1)) + 1e-5
        cam2 = cam2.reshape(b, 1, s2, s2, h // s2, w // s2).permute(0, 1, 2, 4, 3, 5).reshape(b, 1, h, w)
        cam3 = cam3.reshape(b, 1, s3, s3, h // s3, w // s3).permute(0, 1, 2, 4, 3, 5).reshape(b, 1, h, w)

    return cam2, cam3, cam4


def multi_scale_cam_erase(model, inputs_A, inputs_B, scales):
    # cam_list = []
    batch, channel, height, width = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam_a, _cam_b = model(inputs_A_cat, inputs_B_cat, cam_only=True)  # _cam: torch.Size([8, 1, 16, 16])

        _cam_a = F.interpolate(_cam_a, size=(height, width), mode='bilinear', align_corners=False)
        _cam_a = torch.max(_cam_a[:batch, ...], _cam_a[batch:, ...].flip(-1))
        cam_a_list = [F.relu(_cam_a)]

        _cam_b = F.interpolate(_cam_b, size=(height, width), mode='bilinear', align_corners=False)
        _cam_b = torch.max(_cam_b[:batch, ...], _cam_b[batch:, ...].flip(-1))
        cam_b_list = [F.relu(_cam_b)]

        cam_a = torch.sum(torch.stack(cam_a_list, dim=0), dim=0)
        cam_a = cam_a + F.adaptive_max_pool2d(-cam_a, (1, 1))
        cam_a /= F.adaptive_max_pool2d(cam_a, (1, 1)) + 1e-5

        cam_b = torch.sum(torch.stack(cam_b_list, dim=0), dim=0)
        cam_b = cam_b + F.adaptive_max_pool2d(-cam_b, (1, 1))
        cam_b /= F.adaptive_max_pool2d(cam_b, (1, 1)) + 1e-5

    return cam_a, cam_b
