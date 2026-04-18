IMAGE_SOURCE_TABS = ('uov', 'ip', 'inpaint', 'noobai_inpaint', 'noobai_outpaint')
INPAINT_SOURCE_TABS = ('inpaint', 'noobai_inpaint', 'noobai_outpaint')
NOOBAI_INPAINT_SOURCE_TABS = ('noobai_inpaint', 'noobai_outpaint')


def resolve_image_input_selection(input_image_checkbox, current_tab, use_uov, use_ip, use_inpaint,
                                  use_noobai_inpaint, use_noobai_outpaint):
    if not input_image_checkbox:
        return {
            'manual_selection': False,
            'active_tabs': [],
            'active_inpaint_tabs': [],
            'active_inpaint_tab': None,
            'conflict_message': None,
        }

    requested_tabs = []
    for enabled, tab_name in [
        (use_uov, 'uov'),
        (use_ip, 'ip'),
        (use_inpaint, 'inpaint'),
        (use_noobai_inpaint, 'noobai_inpaint'),
        (use_noobai_outpaint, 'noobai_outpaint'),
    ]:
        if enabled:
            requested_tabs.append(tab_name)

    manual_selection = len(requested_tabs) > 0
    active_tabs = requested_tabs.copy() if manual_selection else (
        [current_tab] if current_tab in IMAGE_SOURCE_TABS else []
    )

    active_inpaint_tabs = [tab_name for tab_name in active_tabs if tab_name in INPAINT_SOURCE_TABS]
    non_inpaint_tabs = [tab_name for tab_name in active_tabs if tab_name not in INPAINT_SOURCE_TABS]
    active_inpaint_tab = None
    conflict_message = None

    if len(active_inpaint_tabs) > 1:
        preferred_inpaint_tabs = [
            tab_name for tab_name in active_inpaint_tabs if tab_name in NOOBAI_INPAINT_SOURCE_TABS
        ]
        if len(preferred_inpaint_tabs) == 0:
            preferred_inpaint_tabs = active_inpaint_tabs

        if len(preferred_inpaint_tabs) != len(active_inpaint_tabs):
            ignored_tabs = [
                tab_name for tab_name in active_inpaint_tabs if tab_name not in preferred_inpaint_tabs
            ]
            conflict_message = (
                '[Input] Multiple inpaint tabs were enabled. '
                f'Preferring NoobAI inputs ({", ".join(preferred_inpaint_tabs)}) and ignoring '
                f'{", ".join(ignored_tabs)} for this run.'
            )

        active_inpaint_tabs = preferred_inpaint_tabs
    elif len(active_inpaint_tabs) == 1:
        active_inpaint_tab = active_inpaint_tabs[0]

    if len(active_inpaint_tabs) > 0:
        active_inpaint_tab = current_tab if current_tab in active_inpaint_tabs else active_inpaint_tabs[0]

    active_tabs = non_inpaint_tabs + active_inpaint_tabs

    return {
        'manual_selection': manual_selection,
        'active_tabs': active_tabs,
        'active_inpaint_tabs': active_inpaint_tabs,
        'active_inpaint_tab': active_inpaint_tab,
        'conflict_message': conflict_message,
    }


def merge_noobai_inpaint_inputs(active_inpaint_tabs, noobai_inpaint_input_image, noobai_inpaint_additional_prompt,
                                noobai_outpaint_input_image, noobai_outpaint_additional_prompt,
                                noobai_outpaint_selections):
    use_noobai_inpaint = 'noobai_inpaint' in active_inpaint_tabs
    use_noobai_outpaint = 'noobai_outpaint' in active_inpaint_tabs

    if not use_noobai_inpaint and not use_noobai_outpaint:
        return None

    image = noobai_inpaint_input_image if use_noobai_inpaint else noobai_outpaint_input_image
    if image is None and use_noobai_outpaint:
        image = noobai_outpaint_input_image

    prompts = []
    for prompt in [
        noobai_inpaint_additional_prompt if use_noobai_inpaint else '',
        noobai_outpaint_additional_prompt if use_noobai_outpaint else '',
    ]:
        cleaned_prompt = prompt.strip()
        if cleaned_prompt != '' and cleaned_prompt not in prompts:
            prompts.append(cleaned_prompt)

    message = None
    if use_noobai_inpaint and use_noobai_outpaint:
        message = (
            '[Input] Combining NoobAI inpaint mask and outpaint directions in one run. '
            'The NoobAI Inpaint canvas is used as the base image when it is available.'
        )

    return {
        'image': image,
        'additional_prompt': '\n'.join(prompts),
        'outpaint_selections': list(noobai_outpaint_selections) if use_noobai_outpaint else [],
        'message': message,
    }
