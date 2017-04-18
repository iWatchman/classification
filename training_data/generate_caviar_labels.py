"""Generate frame labels for CAVIAR dataset

CAVIAR dataset provided by the CAVIAR project at http://groups.inf.ed.ac.uk/vision/CAVIAR/CAVIARDATA1/

This file is really concerned with extracting simplified frame labels from the
already existing xml ground truth files provided by the CAVIAR project for the
fighting training data.

Current labels contain a plethora of frame-level data in the following format:
    <frame number="0">

        <objectlist>

            <object id="0">
                <orientation>154</orientation>
                <box xc="133" yc="124" w="43" h="28"/>
                <body>
                    <head xc="116" yc="115" size="4" gaze="325" evaluation="1" occluded="no"/>
                    <shoulders>
                         <right xc="118" yc="124" evaluation="1" occluded="no"/>
                         <left xc="123" yc="113" evaluation="1" occluded="no"/>
                    </shoulders>
                    <hands>
                         <right xc="132" yc="134" size="3" evaluation="1" occluded="no"/>
                         <left xc="140" yc="117" size="3" evaluation="1" occluded="no"/>
                    </hands>
                    <feet>
                         <right xc="152" yc="135" size="3" evaluation="1" occluded="no"/>
                         <left xc="151" yc="129" size="4" evaluation="1" occluded="no"/>
                    </feet>
                </body>
                <appearance>appear</appearance>
                <hypothesislist>
                    <hypothesis id="1" prev="0.0" evaluation="1.0">
                        <movement evaluation="1.0">walking</movement>
                        <role evaluation="1.0">walker</role>
                        <context evaluation="1.0">immobile</context>
                        <situation evaluation="1.0">moving</situation>
                    </hypothesis>
                 </hypothesislist>
            </object>

            ...

        </objectlist>

        <grouplist>

            <group id="0">
                <orientation>156</orientation>
                <box xc="99" yc="128" w="67" h="34"/>
                <members>1,2</members>
                <appearance>appear</appearance>
                <hypothesislist>
                    <hypothesis id="1" prev="0.0" evaluation="1.0">
                        <movement evaluation="1.0">movement</movement>
                        <role evaluation="1.0">fighters</role>
                        <context evaluation="1.0">fighting</context>
                        <situation evaluation="1.0">joining</situation>
                    </hypothesis>
                 </hypothesislist>
            </group>

            ...

        </grouplist>

    </frame>

    ...

    It seems only groups are identified as fighting or not, and this distinction
    can be found in hypothesislist/hypothesis/context. Other appearances of a
    searched-for keyword may appear in hypothesislist/hypothesis/role or
    hypothesislist/hypothesis/situation.

Instead, full frame labels are desired such that each line of the file represents
the label for that frame.
For instance:
    0 |
    0 |=> no violence detected for frames 1-3
    0 |
    1    |
    1    |
    1    |=> violence detected for frames 4-8
    1    |
    1    |
    ...
"""

def get_args():
    '''Get command-line arguments to this script

    Output:
        input: string; directory containing xml files to be parsed
        output: string; directory to save new frame labels
        keyword: string; the keyword to search for in the xml files
    '''

    import argparse
    import os

    class ValidatePathAction(argparse.Action):
        '''Validate given directory or filepaths'''
        def __call__(self, parser, namespace, values, option_string=None):

            # Output directory => may need to be created
            if option_string:
                os.makedirs(values, exist_ok=True)

            # Validate all paths
            if not os.path.exists(values):
                parser.error('Path "{}" does not exist!'.format(values))

            setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser(description='Generate CAVIAR frame labels')
    parser.add_argument('input', type=str, action=ValidatePathAction,
                        help='Directory containing xml files to parse.')
    parser.add_argument('-o', type=str, dest='output', default='.', action=ValidatePathAction,
                        help='Directory to save generated frame labels. Defaults to current working directory.')
    parser.add_argument('-k', type=str, dest='keyword', default='fighting',
                        help='Keyword to search in xml files. Defaults to "fighting".')

    return parser.parse_args()

def find_key_frames(filename, keyword):
    '''Find all frames where keyword is found in xml file

    Input:
        filename: string; name of xml file to parse
        keyword: string; keyword to search for`

    Output:
        key_frames: set; all frame numbers containing keyword
        total: int; total number of frames
    '''

    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    hypothesis_children = ['role', 'context', 'situation']
    key_frames = set()

    # In the xml files, hypothesis on groups and objects have four children:
    # movement, role, context, and situation. From examination of the xml files,
    # it is apparent that keywords may appear in any of the last three children,
    # thus each needs to be examined separately and results combined into one set.
    #
    # For help in tree.findall("..."), see python documentation for XPath support
    # of ElementTree XML API
    for child in hypothesis_children:
        frames = tree.findall(".//hypothesis[{}='{}']/../../../..".format(child, keyword))
        key_frames.update([int(f.attrib['number']) for f in frames])

    # NOTE: Frames are numbered beginning with 0 in these xml files
    return (key_frames, 1 + int(tree.find("./frame[last()]").attrib['number']))

def main():
    '''Main script logic'''

    import glob
    import os

    args = get_args()
    xmls = glob.glob(os.path.join(args.input, '*.xml'))

    for xml in xmls:
        basename = os.path.splitext(os.path.basename(xml))[0]
        key_frames, total = find_key_frames(xml, args.keyword)

        with open(os.path.join(args.output, '{}_labels.txt'.format(basename)), 'w') as f:
            for i in range(total):
                if i in key_frames:
                    f.write('1\n')
                else:
                    f.write('0\n')

if __name__ == '__main__':
    main()
